#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace photon::cuda_apriltag {

struct Calibration {
  double fx, fy, cx, cy, k1, k2, p1, p2, k3;
};

struct GrayFrame {
  const std::uint8_t* data;
  int width;
  int height;
  std::size_t stride_bytes;
};

struct Detection {
  std::string family;
  int id;
  int hamming;
  float decision_margin;
  std::array<double, 9> homography;
  double center_x;
  double center_y;
  std::array<double, 8> corners;
};

class DetectorBackend {
 public:
  virtual ~DetectorBackend() = default;
  virtual void SetCalibration(const Calibration& calibration) = 0;
  virtual std::vector<Detection> Process(const GrayFrame& frame) = 0;
};

using DetectorFactory = std::function<std::unique_ptr<DetectorBackend>(
    int width, int height, int decimate)>;

struct DetectorSlot;

class DetectorRegistry {
 public:
  explicit DetectorRegistry(DetectorFactory factory);

  std::int64_t Create(int width, int height, int decimate);
  void SetCalibration(std::int64_t handle, const Calibration& calibration);
  std::vector<Detection> Process(std::int64_t handle, const GrayFrame& frame);
  void Destroy(std::int64_t handle);
  std::size_t Size() const;

 private:
  std::shared_ptr<DetectorSlot> FindSlot(std::int64_t handle) const;

  DetectorFactory factory_;
  mutable std::mutex registry_mutex_;
  std::unordered_map<std::int64_t, std::shared_ptr<DetectorSlot>> slots_;
  std::int64_t next_handle_ = 1;
};

}  // namespace photon::cuda_apriltag
