#pragma once

#include "DetectorRegistry.h"

#include <cstdint>
#include <memory>
#include <vector>

struct apriltag_detector;
struct apriltag_family;

namespace frc971::apriltag {
class GpuDetector;
}

namespace photon::cuda_apriltag {

class CudaAprilTagDetector final : public DetectorBackend {
 public:
  CudaAprilTagDetector(int width, int height, int decimate);
  ~CudaAprilTagDetector() override;

  CudaAprilTagDetector(const CudaAprilTagDetector&) = delete;
  CudaAprilTagDetector& operator=(const CudaAprilTagDetector&) = delete;

  void SetCalibration(const Calibration& calibration) override;
  std::vector<Detection> Process(const GrayFrame& frame) override;

 private:
  struct TagFamilyDeleter {
    void operator()(apriltag_family* family) const noexcept;
  };
  struct TagDetectorDeleter {
    void operator()(apriltag_detector* detector) const noexcept;
  };

  int width_;
  int height_;
  int decimate_;
  std::vector<std::uint8_t> packed_;
  std::unique_ptr<apriltag_family, TagFamilyDeleter> tag_family_;
  std::unique_ptr<apriltag_detector, TagDetectorDeleter> tag_detector_;
  std::unique_ptr<frc971::apriltag::GpuDetector> gpu_detector_;
};

}  // namespace photon::cuda_apriltag
