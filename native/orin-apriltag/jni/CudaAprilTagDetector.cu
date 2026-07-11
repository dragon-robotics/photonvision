#include "CudaAprilTagDetector.h"

#include "frc971/orin/971apriltag.h"
#include "third_party/apriltag/apriltag.h"
#include "third_party/apriltag/common/matd.h"
#include "third_party/apriltag/common/workerpool.h"
#include "third_party/apriltag/common/zarray.h"
#include "third_party/apriltag/tag36h11.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace photon::cuda_apriltag {
namespace {

constexpr int kDetectorThreads = 6;
constexpr std::size_t kVendorMaximumPixels = 1U << 22;

std::size_t CheckedImageSize(int width, int height) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("Detector dimensions must be positive");
  }
  const auto unsigned_width = static_cast<std::size_t>(width);
  const auto unsigned_height = static_cast<std::size_t>(height);
  if (unsigned_width >
      std::numeric_limits<std::size_t>::max() / unsigned_height) {
    throw std::overflow_error("Detector image size overflows size_t");
  }
  return unsigned_width * unsigned_height;
}

void ValidateDetectorConfiguration(int width, int height, int decimate) {
  const std::size_t pixels = CheckedImageSize(width, height);
  if (decimate != 1 && decimate != 2) {
    throw std::invalid_argument("CUDA detector decimation must be 1 or 2");
  }
  if (width % 8 != 0 || height % 8 != 0) {
    throw std::invalid_argument(
        "CUDA detector dimensions must be divisible by 8");
  }
  if (width / decimate <= 2 || height / decimate <= 2) {
    throw std::invalid_argument("CUDA detector dimensions are too small");
  }
  if (pixels >= kVendorMaximumPixels) {
    throw std::invalid_argument(
        "CUDA detector image must contain fewer than 4194304 pixels");
  }
}

void ValidateCalibration(const Calibration& calibration) {
  const double values[] = {
      calibration.fx, calibration.fy, calibration.cx,
      calibration.cy, calibration.k1, calibration.k2,
      calibration.p1, calibration.p2, calibration.k3,
  };
  for (double value : values) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("Calibration values must be finite");
    }
  }
  if (calibration.fx <= 0.0 || calibration.fy <= 0.0) {
    throw std::invalid_argument(
        "Calibration focal lengths must be positive");
  }
}

void ValidateFrame(const GrayFrame& frame, int width, int height) {
  if (frame.data == nullptr) {
    throw std::invalid_argument("Gray frame data must not be null");
  }
  if (frame.width <= 0 || frame.height <= 0) {
    throw std::invalid_argument("Gray frame dimensions must be positive");
  }
  if (frame.width != width || frame.height != height) {
    throw std::invalid_argument(
        "Gray frame dimensions do not match detector configuration");
  }
  const auto row_bytes = static_cast<std::size_t>(frame.width);
  if (frame.stride_bytes < row_bytes) {
    throw std::invalid_argument("Gray frame stride must be at least its width");
  }
  const auto rows_before_last = static_cast<std::size_t>(frame.height - 1);
  if (rows_before_last >
      (std::numeric_limits<std::size_t>::max() - row_bytes) /
          frame.stride_bytes) {
    throw std::overflow_error("Gray frame buffer size overflows size_t");
  }
  const std::size_t span_bytes =
      rows_before_last * frame.stride_bytes + row_bytes;
  if (span_bytes > static_cast<std::size_t>(
                       std::numeric_limits<std::ptrdiff_t>::max())) {
    throw std::overflow_error("Gray frame span exceeds PTRDIFF_MAX");
  }
  const std::size_t last_offset = span_bytes - 1;
  if (last_offset > std::numeric_limits<std::uintptr_t>::max()) {
    throw std::overflow_error("Gray frame address range exceeds UINTPTR_MAX");
  }
  const auto base = reinterpret_cast<std::uintptr_t>(frame.data);
  if (base > std::numeric_limits<std::uintptr_t>::max() -
                 static_cast<std::uintptr_t>(last_offset)) {
    throw std::overflow_error("Gray frame address range exceeds UINTPTR_MAX");
  }
}

Detection CopyDetection(const apriltag_detection_t& native) {
  if (native.family == nullptr || native.family->name == nullptr ||
      native.H == nullptr || native.H->nrows != 3 || native.H->ncols != 3) {
    throw std::runtime_error("CUDA detector returned an incomplete detection");
  }

  Detection result{};
  result.family = native.family->name;
  result.id = native.id;
  result.hamming = native.hamming;
  result.decision_margin = native.decision_margin;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      result.homography[static_cast<std::size_t>(row * 3 + column)] =
          MATD_EL(native.H, row, column);
    }
  }
  result.center_x = native.c[0];
  result.center_y = native.c[1];
  for (int corner = 0; corner < 4; ++corner) {
    result.corners[static_cast<std::size_t>(corner * 2)] = native.p[corner][0];
    result.corners[static_cast<std::size_t>(corner * 2 + 1)] =
        native.p[corner][1];
  }
  return result;
}

}  // namespace

void CudaAprilTagDetector::TagFamilyDeleter::operator()(
    apriltag_family* family) const noexcept {
  tag36h11_destroy(family);
}

void CudaAprilTagDetector::TagDetectorDeleter::operator()(
    apriltag_detector* detector) const noexcept {
  apriltag_detector_destroy(detector);
}

CudaAprilTagDetector::CudaAprilTagDetector(int width, int height, int decimate)
    : width_(width), height_(height), decimate_(decimate) {
  ValidateDetectorConfiguration(width_, height_, decimate_);

  tag_family_.reset(tag36h11_create());
  if (!tag_family_) {
    throw std::runtime_error("Failed to create tag36h11 family");
  }

  tag_detector_.reset(apriltag_detector_create());
  if (!tag_detector_) {
    throw std::runtime_error("Failed to create AprilTag detector");
  }
  tag_detector_->quad_decimate = static_cast<float>(decimate_);
  tag_detector_->nthreads = kDetectorThreads;
  tag_detector_->wp = workerpool_create(kDetectorThreads);
  if (tag_detector_->wp == nullptr) {
    throw std::runtime_error("Failed to create AprilTag worker pool");
  }
  apriltag_detector_add_family(tag_detector_.get(), tag_family_.get());
}

CudaAprilTagDetector::~CudaAprilTagDetector() = default;

void CudaAprilTagDetector::SetCalibration(const Calibration& calibration) {
  gpu_detector_.reset();
  ValidateCalibration(calibration);

  const frc971::apriltag::CameraMatrix camera_matrix{
      .fx = calibration.fx,
      .cx = calibration.cx,
      .fy = calibration.fy,
      .cy = calibration.cy,
  };
  const frc971::apriltag::DistCoeffs distortion{
      .k1 = calibration.k1,
      .k2 = calibration.k2,
      .p1 = calibration.p1,
      .p2 = calibration.p2,
      .k3 = calibration.k3,
  };
  gpu_detector_ = std::make_unique<frc971::apriltag::GpuDetector>(
      static_cast<std::size_t>(width_), static_cast<std::size_t>(height_),
      tag_detector_.get(), camera_matrix, distortion,
      static_cast<std::size_t>(decimate_));
}

std::vector<Detection> CudaAprilTagDetector::Process(const GrayFrame& frame) {
  ValidateFrame(frame, width_, height_);
  if (!gpu_detector_) {
    throw std::runtime_error(
        "CUDA detector calibration is not configured");
  }

  const std::uint8_t* input = frame.data;
  if (frame.stride_bytes != static_cast<std::size_t>(frame.width)) {
    packed_.resize(CheckedImageSize(frame.width, frame.height));
    for (int row = 0; row < frame.height; ++row) {
      std::memcpy(
          packed_.data() + static_cast<std::size_t>(row) * frame.width,
          frame.data + static_cast<std::size_t>(row) * frame.stride_bytes,
          static_cast<std::size_t>(frame.width));
    }
    input = packed_.data();
  }

  gpu_detector_->DetectGrayHost(const_cast<std::uint8_t*>(input));
  const zarray_t* native_detections = gpu_detector_->Detections();
  if (native_detections == nullptr) {
    throw std::runtime_error("CUDA detector returned a null detection array");
  }

  std::vector<Detection> detections;
  const int count = zarray_size(native_detections);
  detections.reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    apriltag_detection_t* native = nullptr;
    zarray_get(native_detections, index, &native);
    if (native == nullptr) {
      throw std::runtime_error("CUDA detector returned a null detection");
    }
    detections.push_back(CopyDetection(*native));
  }
  return detections;
}

}  // namespace photon::cuda_apriltag
