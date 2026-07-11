#include "CudaAprilTagDetector.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
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

}  // namespace

int main() {
  try {
    constexpr int kWidth = 1280;
    constexpr int kHeight = 800;
    constexpr int kDecimate = 2;
    constexpr int kFrameCount = 120;
    constexpr std::size_t kAllowedDelta = 32ULL * 1024ULL * 1024ULL;

    if (::setenv("CUDA_MODULE_LOADING", "EAGER", 1) != 0) {
      throw std::runtime_error("Failed to request eager CUDA module loading");
    }
    TEST_CHECK_CUDA(cudaFree(nullptr));
    TEST_CHECK_CUDA(cudaDeviceSynchronize());

    std::size_t baseline_free = 0;
    std::size_t baseline_total = 0;
    TEST_CHECK_CUDA(cudaMemGetInfo(&baseline_free, &baseline_total));

    std::vector<std::uint8_t> pixels(
        static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(kHeight),
        0);
    {
      photon::cuda_apriltag::CudaAprilTagDetector detector(kWidth, kHeight,
                                                            kDecimate);
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
      const auto process_end = std::chrono::steady_clock::now();
      const auto runtime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  process_end - process_start)
                                  .count();
      std::cout << "Processed " << kFrameCount << " frames in " << runtime_ms
                << " ms\n";
    }

    TEST_CHECK_CUDA(cudaDeviceSynchronize());
    std::size_t after_free = 0;
    std::size_t after_total = 0;
    TEST_CHECK_CUDA(cudaMemGetInfo(&after_free, &after_total));

    const std::size_t delta = baseline_free > after_free
                                  ? baseline_free - after_free
                                  : after_free - baseline_free;
    std::cout << "CUDA memory baseline_free=" << baseline_free
              << " after_free=" << after_free << " total=" << after_total
              << " absolute_delta=" << delta << '\n';
    if (baseline_total != after_total) {
      throw std::runtime_error("CUDA total memory changed during smoke test");
    }
    if (delta > kAllowedDelta) {
      throw std::runtime_error("CUDA free-memory delta exceeded 32 MiB");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CUDA detector smoke test failed: " << error.what() << '\n';
    return 1;
  }
}
