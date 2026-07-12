#include "CudaAprilTagDetector.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

void CheckCuda(cudaError_t status, const char* expression) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string("CUDA failure: ") + expression + ": " +
                           cudaGetErrorString(status));
}

#define TEST_CHECK_CUDA(expression) CheckCuda((expression), #expression)

void CheckCudaDriver(CUresult status, const char* expression) {
  if (status == CUDA_SUCCESS) {
    return;
  }
  const char* message = nullptr;
  cuGetErrorString(status, &message);
  throw std::runtime_error(std::string("CUDA driver failure: ") + expression +
                           ": " +
                           (message == nullptr ? "unknown error" : message));
}

#define TEST_CHECK_CUDA_DRIVER(expression) \
  CheckCudaDriver((expression), #expression)

struct TrackedCudaResources {
  std::unordered_map<void*, std::size_t> device_allocations;
  std::unordered_map<void*, std::size_t> pinned_host_allocations;
  std::unordered_set<void*> streams;
  std::unordered_set<void*> events;
};

struct TrackedCudaActivity {
  std::size_t device_allocation_calls = 0;
  std::size_t pinned_host_allocation_calls = 0;
  std::size_t stream_creation_calls = 0;
  std::size_t event_creation_calls = 0;
};

constexpr std::size_t kMeasuredLifecycleCount = 5;

struct FreeMemorySample {
  std::size_t before;
  std::size_t after;
};

std::size_t DirectionalRetainedDelta(const FreeMemorySample& sample) {
  return sample.before > sample.after ? sample.before - sample.after : 0;
}

std::size_t MedianDirectionalRetainedDelta(
    const std::array<FreeMemorySample, kMeasuredLifecycleCount>& samples) {
  std::array<std::size_t, kMeasuredLifecycleCount> retained_deltas{};
  for (std::size_t index = 0; index < samples.size(); ++index) {
    retained_deltas[index] = DirectionalRetainedDelta(samples[index]);
  }
  std::sort(retained_deltas.begin(), retained_deltas.end());
  return retained_deltas[retained_deltas.size() / 2];
}

void TestMedianDirectionalRetainedDelta() {
  const std::array<FreeMemorySample, kMeasuredLifecycleCount> samples{{
      {.before = 100, .after = 90},
      {.before = 100, .after = 130},
      {.before = 100, .after = 70},
      {.before = 100, .after = 50},
      {.before = 100, .after = 80},
  }};
  constexpr std::size_t kExpectedMedian = 20;
  const std::size_t actual_median =
      MedianDirectionalRetainedDelta(samples);
  if (actual_median != kExpectedMedian) {
    throw std::runtime_error(
        "Directional retained-delta median helper expected 20, got " +
        std::to_string(actual_median));
  }
}

std::mutex resource_mutex;
std::unordered_map<void*, std::size_t> device_allocations;
std::unordered_map<void*, std::size_t> pinned_host_allocations;
std::unordered_set<void*> streams;
std::unordered_set<void*> events;
TrackedCudaActivity tracked_activity;

TrackedCudaResources SnapshotTrackedResources() {
  std::scoped_lock lock(resource_mutex);
  return {
      .device_allocations = device_allocations,
      .pinned_host_allocations = pinned_host_allocations,
      .streams = streams,
      .events = events,
  };
}

TrackedCudaActivity SnapshotTrackedActivity() {
  std::scoped_lock lock(resource_mutex);
  return tracked_activity;
}

std::string ResourceCounts(const TrackedCudaResources& resources) {
  std::ostringstream message;
  message << "device_allocations=" << resources.device_allocations.size()
          << " pinned_host_allocations="
          << resources.pinned_host_allocations.size()
          << " streams=" << resources.streams.size()
          << " events=" << resources.events.size();
  return message.str();
}

void RequireResourcesRestored(const TrackedCudaResources& before,
                              const TrackedCudaResources& after,
                              const std::string& lifecycle) {
  if (before.device_allocations != after.device_allocations ||
      before.pinned_host_allocations != after.pinned_host_allocations ||
      before.streams != after.streams || before.events != after.events) {
    throw std::runtime_error(
        "CUDA RAII resources were retained after " + lifecycle +
        ": before " + ResourceCounts(before) + "; after " +
        ResourceCounts(after));
  }
}

void RequireWarmupActivity(const TrackedCudaActivity& before,
                           const TrackedCudaActivity& after) {
  if (after.device_allocation_calls <= before.device_allocation_calls ||
      after.pinned_host_allocation_calls <=
          before.pinned_host_allocation_calls ||
      after.stream_creation_calls <= before.stream_creation_calls ||
      after.event_creation_calls <= before.event_creation_calls) {
    throw std::runtime_error(
        "Warmup did not exercise every wrapped CUDA resource category");
  }
}

template <typename Callable>
void ExpectOverflow(const char* expected_message, Callable callable) {
  try {
    callable();
  } catch (const std::overflow_error& error) {
    if (error.what() == std::string(expected_message)) {
      return;
    }
    throw std::runtime_error(std::string("Unexpected overflow error: ") +
                             error.what());
  } catch (const std::exception& error) {
    throw std::runtime_error(std::string("Expected overflow error, got: ") +
                             error.what());
  }
  throw std::runtime_error("Expected overflow error was not thrown");
}

}  // namespace

extern "C" {

cudaError_t __real_cudaMalloc(void** pointer, std::size_t bytes);
cudaError_t __real_cudaFree(void* pointer);
cudaError_t __real_cudaMallocHost(void** pointer, std::size_t bytes);
cudaError_t __real_cudaFreeHost(void* pointer);
cudaError_t __real_cudaStreamCreate(cudaStream_t* stream);
cudaError_t __real_cudaStreamDestroy(cudaStream_t stream);
cudaError_t __real_cudaEventCreate(cudaEvent_t* event);
cudaError_t __real_cudaEventDestroy(cudaEvent_t event);

cudaError_t __wrap_cudaMalloc(void** pointer, std::size_t bytes) {
  const cudaError_t status = __real_cudaMalloc(pointer, bytes);
  if (status == cudaSuccess && pointer != nullptr && *pointer != nullptr) {
    std::scoped_lock lock(resource_mutex);
    device_allocations[*pointer] = bytes;
    ++tracked_activity.device_allocation_calls;
  }
  return status;
}

cudaError_t __wrap_cudaFree(void* pointer) {
  const cudaError_t status = __real_cudaFree(pointer);
  if (status == cudaSuccess && pointer != nullptr) {
    std::scoped_lock lock(resource_mutex);
    device_allocations.erase(pointer);
  }
  return status;
}

cudaError_t __wrap_cudaMallocHost(void** pointer, std::size_t bytes) {
  const cudaError_t status = __real_cudaMallocHost(pointer, bytes);
  if (status == cudaSuccess && pointer != nullptr && *pointer != nullptr) {
    std::scoped_lock lock(resource_mutex);
    pinned_host_allocations[*pointer] = bytes;
    ++tracked_activity.pinned_host_allocation_calls;
  }
  return status;
}

cudaError_t __wrap_cudaFreeHost(void* pointer) {
  const cudaError_t status = __real_cudaFreeHost(pointer);
  if (status == cudaSuccess && pointer != nullptr) {
    std::scoped_lock lock(resource_mutex);
    pinned_host_allocations.erase(pointer);
  }
  return status;
}

cudaError_t __wrap_cudaStreamCreate(cudaStream_t* stream) {
  const cudaError_t status = __real_cudaStreamCreate(stream);
  if (status == cudaSuccess && stream != nullptr && *stream != nullptr) {
    std::scoped_lock lock(resource_mutex);
    streams.insert(static_cast<void*>(*stream));
    ++tracked_activity.stream_creation_calls;
  }
  return status;
}

cudaError_t __wrap_cudaStreamDestroy(cudaStream_t stream) {
  const cudaError_t status = __real_cudaStreamDestroy(stream);
  if (status == cudaSuccess && stream != nullptr) {
    std::scoped_lock lock(resource_mutex);
    streams.erase(static_cast<void*>(stream));
  }
  return status;
}

cudaError_t __wrap_cudaEventCreate(cudaEvent_t* event) {
  const cudaError_t status = __real_cudaEventCreate(event);
  if (status == cudaSuccess && event != nullptr && *event != nullptr) {
    std::scoped_lock lock(resource_mutex);
    events.insert(static_cast<void*>(*event));
    ++tracked_activity.event_creation_calls;
  }
  return status;
}

cudaError_t __wrap_cudaEventDestroy(cudaEvent_t event) {
  const cudaError_t status = __real_cudaEventDestroy(event);
  if (status == cudaSuccess && event != nullptr) {
    std::scoped_lock lock(resource_mutex);
    events.erase(static_cast<void*>(event));
  }
  return status;
}

}  // extern "C"

int main() {
  try {
    constexpr int kWidth = 1280;
    constexpr int kHeight = 800;
    constexpr int kDecimate = 2;
    constexpr int kFrameCount = 120;
    constexpr std::size_t kAllowedRetainedDelta =
        32ULL * 1024ULL * 1024ULL;

    TestMedianDirectionalRetainedDelta();

    if (::setenv("CUDA_MODULE_LOADING", "EAGER", 1) != 0) {
      throw std::runtime_error("Failed to request eager CUDA module loading");
    }
    TEST_CHECK_CUDA(cudaFree(nullptr));
    TEST_CHECK_CUDA(cudaDeviceSynchronize());
    CUmoduleLoadingMode module_loading_mode{};
    TEST_CHECK_CUDA_DRIVER(cuModuleGetLoadingMode(&module_loading_mode));
    if (module_loading_mode != CU_MODULE_EAGER_LOADING) {
      throw std::runtime_error("CUDA module loading mode is not EAGER");
    }

    std::vector<std::uint8_t> pixels(
        static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(kHeight),
        0);
    constexpr std::size_t kPaddedStride = kWidth + 32;
    std::vector<std::uint8_t> padded_pixels(
        kPaddedStride * static_cast<std::size_t>(kHeight), 0);
    const auto run_detector_lifecycle = [&] {
      photon::cuda_apriltag::CudaAprilTagDetector detector(kWidth, kHeight,
                                                            kDecimate);
      const auto maximum_pointer_span = static_cast<std::size_t>(
          std::numeric_limits<std::ptrdiff_t>::max());
      const auto rows_before_last = static_cast<std::size_t>(kHeight - 1);
      const std::size_t oversized_stride =
          (maximum_pointer_span - static_cast<std::size_t>(kWidth)) /
              rows_before_last +
          1;
      const photon::cuda_apriltag::GrayFrame oversized_span_frame{
          .data = pixels.data(),
          .width = kWidth,
          .height = kHeight,
          .stride_bytes = oversized_stride,
      };
      ExpectOverflow("Gray frame span exceeds PTRDIFF_MAX", [&] {
        detector.Process(oversized_span_frame);
      });

      constexpr std::size_t kContiguousSpan =
          static_cast<std::size_t>(kWidth) * kHeight;
      const auto near_maximum_address =
          std::numeric_limits<std::uintptr_t>::max() - (kContiguousSpan - 2);
      const photon::cuda_apriltag::GrayFrame wrapped_address_frame{
          .data = reinterpret_cast<const std::uint8_t*>(near_maximum_address),
          .width = kWidth,
          .height = kHeight,
          .stride_bytes = static_cast<std::size_t>(kWidth),
      };
      ExpectOverflow("Gray frame address range exceeds UINTPTR_MAX", [&] {
        detector.Process(wrapped_address_frame);
      });

      detector.SetCalibration({
          .fx = 1000.0,
          .fy = 1000.0,
          .cx = kWidth / 2.0,
          .cy = kHeight / 2.0,
          .k1 = 0.0,
          .k2 = 0.0,
          .p1 = 0.0,
          .p2 = 0.0,
          .k3 = 0.0,
      });
      const photon::cuda_apriltag::GrayFrame frame{
          .data = pixels.data(),
          .width = kWidth,
          .height = kHeight,
          .stride_bytes = static_cast<std::size_t>(kWidth),
      };
      const auto process_start = std::chrono::steady_clock::now();
      for (int frame_index = 0; frame_index < kFrameCount; ++frame_index) {
        const auto detections = detector.Process(frame);
        if (!detections.empty()) {
          throw std::runtime_error("Black frame unexpectedly produced a detection");
        }
      }
      const photon::cuda_apriltag::GrayFrame padded_frame{
          .data = padded_pixels.data(),
          .width = kWidth,
          .height = kHeight,
          .stride_bytes = kPaddedStride,
      };
      if (!detector.Process(padded_frame).empty()) {
        throw std::runtime_error(
            "Padded-stride black frame unexpectedly produced a detection");
      }
      const auto process_end = std::chrono::steady_clock::now();
      const auto runtime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  process_end - process_start)
                                  .count();
      std::cout << "Processed " << kFrameCount << " frames in " << runtime_ms
                << " ms plus one padded-stride frame\n";
    };

    const TrackedCudaResources warmup_resources_before =
        SnapshotTrackedResources();
    const TrackedCudaActivity warmup_activity_before =
        SnapshotTrackedActivity();
    run_detector_lifecycle();
    TEST_CHECK_CUDA(cudaDeviceSynchronize());
    const TrackedCudaResources warmup_resources_after =
        SnapshotTrackedResources();
    const TrackedCudaActivity warmup_activity_after =
        SnapshotTrackedActivity();
    RequireResourcesRestored(warmup_resources_before, warmup_resources_after,
                             "warmup destroy");
    RequireWarmupActivity(warmup_activity_before, warmup_activity_after);
    std::cout << "CUDA warmup activity device_allocations="
              << warmup_activity_after.device_allocation_calls -
                     warmup_activity_before.device_allocation_calls
              << " pinned_host_allocations="
              << warmup_activity_after.pinned_host_allocation_calls -
                     warmup_activity_before.pinned_host_allocation_calls
              << " streams="
              << warmup_activity_after.stream_creation_calls -
                     warmup_activity_before.stream_creation_calls
              << " events="
              << warmup_activity_after.event_creation_calls -
                     warmup_activity_before.event_creation_calls
              << '\n';

    std::size_t baseline_free = 0;
    std::size_t baseline_total = 0;
    TEST_CHECK_CUDA(cudaMemGetInfo(&baseline_free, &baseline_total));
    std::cout << "CUDA warmed baseline_free=" << baseline_free
              << " total=" << baseline_total << '\n';

    std::array<FreeMemorySample, kMeasuredLifecycleCount> free_memory_samples{};
    for (std::size_t cycle = 0; cycle < kMeasuredLifecycleCount; ++cycle) {
      std::size_t cycle_before_free = 0;
      std::size_t cycle_before_total = 0;
      TEST_CHECK_CUDA(
          cudaMemGetInfo(&cycle_before_free, &cycle_before_total));
      if (cycle_before_total != baseline_total) {
        throw std::runtime_error("CUDA total memory changed during smoke test");
      }
      const TrackedCudaResources resources_before =
          SnapshotTrackedResources();

      run_detector_lifecycle();
      TEST_CHECK_CUDA(cudaDeviceSynchronize());
      const TrackedCudaResources resources_after = SnapshotTrackedResources();
      RequireResourcesRestored(resources_before, resources_after,
                               "measured lifecycle " +
                                   std::to_string(cycle) + " destroy");
      std::cout << "CUDA tracked resources cycle=" << cycle
                << " device_allocations="
                << resources_after.device_allocations.size()
                << " pinned_host_allocations="
                << resources_after.pinned_host_allocations.size()
                << " streams=" << resources_after.streams.size()
                << " events=" << resources_after.events.size() << '\n';

      std::size_t cycle_after_free = 0;
      std::size_t cycle_after_total = 0;
      TEST_CHECK_CUDA(cudaMemGetInfo(&cycle_after_free, &cycle_after_total));
      if (cycle_after_total != baseline_total) {
        throw std::runtime_error("CUDA total memory changed during smoke test");
      }
      free_memory_samples[cycle] = {
          .before = cycle_before_free,
          .after = cycle_after_free,
      };
      const std::size_t retained_delta =
          DirectionalRetainedDelta(free_memory_samples[cycle]);
      std::cout << "CUDA lifecycle cycle=" << cycle
                << " before_free=" << cycle_before_free
                << " after_free=" << cycle_after_free
                << " retained_delta=" << retained_delta
                << " total=" << cycle_after_total << '\n';
    }

    const std::size_t median_retained_delta =
        MedianDirectionalRetainedDelta(free_memory_samples);
    std::cout << "CUDA median directional retained_delta="
              << median_retained_delta << " limit=" << kAllowedRetainedDelta
              << '\n';
    if (median_retained_delta >= kAllowedRetainedDelta) {
      throw std::runtime_error(
          "CUDA median directional retained delta must be less than 32 MiB");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CUDA detector smoke test failed: " << error.what() << '\n';
    return 1;
  }
}
