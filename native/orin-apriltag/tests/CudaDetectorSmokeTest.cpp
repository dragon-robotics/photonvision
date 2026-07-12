#include "CudaAprilTagDetector.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <mutex>
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

std::mutex resource_mutex;
std::unordered_map<void*, std::size_t> device_allocations;
std::unordered_map<void*, std::size_t> pinned_host_allocations;
std::unordered_set<void*> streams;
std::unordered_set<void*> events;

TrackedCudaResources SnapshotTrackedResources() {
  std::scoped_lock lock(resource_mutex);
  return {
      .device_allocations = device_allocations,
      .pinned_host_allocations = pinned_host_allocations,
      .streams = streams,
      .events = events,
  };
}

void RequireResourcesRestored(const TrackedCudaResources& before,
                              const TrackedCudaResources& after) {
  if (before.device_allocations != after.device_allocations ||
      before.pinned_host_allocations != after.pinned_host_allocations ||
      before.streams != after.streams || before.events != after.events) {
    throw std::runtime_error("CUDA RAII resources were retained after destroy");
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
    constexpr int kMeasuredLifecycleCount = 5;

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

    run_detector_lifecycle();
    TEST_CHECK_CUDA(cudaDeviceSynchronize());
    std::size_t baseline_free = 0;
    std::size_t baseline_total = 0;
    TEST_CHECK_CUDA(cudaMemGetInfo(&baseline_free, &baseline_total));
    std::cout << "CUDA warmed baseline_free=" << baseline_free
              << " total=" << baseline_total << '\n';

    for (int cycle = 0; cycle < kMeasuredLifecycleCount; ++cycle) {
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
      RequireResourcesRestored(resources_before, resources_after);
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
      std::cout << "CUDA lifecycle cycle=" << cycle
                << " before_free=" << cycle_before_free
                << " after_free=" << cycle_after_free
                << " total=" << cycle_after_total << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CUDA detector smoke test failed: " << error.what() << '\n';
    return 1;
  }
}
