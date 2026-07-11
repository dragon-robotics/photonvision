#include "DetectorRegistry.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace photon::cuda_apriltag {

namespace {

constexpr char kInvalidHandleMessage[] = "Invalid CUDA detector handle";

}  // namespace

struct DetectorSlot {
  std::mutex mutex;
  std::unique_ptr<DetectorBackend> detector;
};

DetectorRegistry::DetectorRegistry(DetectorFactory factory)
    : factory_(std::move(factory)) {}

std::int64_t DetectorRegistry::Create(int width, int height, int decimate) {
  auto detector = factory_(width, height, decimate);
  if (!detector) {
    throw std::runtime_error("Detector factory returned null");
  }

  auto slot = std::make_shared<DetectorSlot>();
  slot->detector = std::move(detector);

  std::scoped_lock lock(registry_mutex_);
  if (next_handle_ == std::numeric_limits<std::int64_t>::max()) {
    throw std::overflow_error("CUDA detector handle space exhausted");
  }
  const std::int64_t handle = next_handle_;
  ++next_handle_;
  slots_.emplace(handle, std::move(slot));
  return handle;
}

void DetectorRegistry::SetCalibration(std::int64_t handle,
                                      const Calibration& calibration) {
  const auto slot = FindSlot(handle);
  std::scoped_lock lock(slot->mutex);
  if (!slot->detector) {
    throw std::invalid_argument(kInvalidHandleMessage);
  }
  slot->detector->SetCalibration(calibration);
}

std::vector<Detection> DetectorRegistry::Process(std::int64_t handle,
                                                  const GrayFrame& frame) {
  const auto slot = FindSlot(handle);
  std::scoped_lock lock(slot->mutex);
  if (!slot->detector) {
    throw std::invalid_argument(kInvalidHandleMessage);
  }
  return slot->detector->Process(frame);
}

void DetectorRegistry::Destroy(std::int64_t handle) {
  std::shared_ptr<DetectorSlot> slot;
  {
    std::scoped_lock lock(registry_mutex_);
    const auto it = slots_.find(handle);
    if (it == slots_.end()) {
      throw std::invalid_argument(kInvalidHandleMessage);
    }
    slot = std::move(it->second);
    slots_.erase(it);
  }

  std::scoped_lock lock(slot->mutex);
  slot->detector.reset();
}

std::size_t DetectorRegistry::Size() const {
  std::scoped_lock lock(registry_mutex_);
  return slots_.size();
}

std::shared_ptr<DetectorSlot> DetectorRegistry::FindSlot(
    std::int64_t handle) const {
  std::scoped_lock lock(registry_mutex_);
  const auto it = slots_.find(handle);
  if (it == slots_.end()) {
    throw std::invalid_argument(kInvalidHandleMessage);
  }
  return it->second;
}

}  // namespace photon::cuda_apriltag
