#include "DetectorRegistry.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace {

using photon::cuda_apriltag::Calibration;
using photon::cuda_apriltag::Detection;
using photon::cuda_apriltag::DetectorBackend;
using photon::cuda_apriltag::DetectorRegistry;
using photon::cuda_apriltag::GrayFrame;

using namespace std::chrono_literals;

constexpr auto kTimeout = 1s;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Callable>
void RequireInvalidHandle(Callable&& callable, const std::string& operation) {
  try {
    callable();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(operation + " did not reject an invalid handle");
}

template <typename Callable>
bool RejectsInvalidHandle(Callable&& callable) {
  try {
    callable();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

void CaptureFuture(std::future<void>& future,
                   std::exception_ptr& first_error) {
  if (!future.valid()) {
    return;
  }
  try {
    future.get();
  } catch (...) {
    if (!first_error) {
      first_error = std::current_exception();
    }
  }
}

template <typename T>
std::optional<T> CaptureFuture(std::future<T>& future,
                               std::exception_ptr& first_error) {
  if (!future.valid()) {
    return std::nullopt;
  }
  try {
    return future.get();
  } catch (...) {
    if (!first_error) {
      first_error = std::current_exception();
    }
    return std::nullopt;
  }
}

GrayFrame Frame() {
  static const std::uint8_t pixels[] = {0};
  return {pixels, 1, 1, 1};
}

Calibration TestCalibration() {
  return {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
}

class RecordingBackend final : public DetectorBackend {
 public:
  void SetCalibration(const Calibration& calibration) override {
    calibration_ = calibration;
    calibration_calls_++;
  }

  std::vector<Detection> Process(const GrayFrame&) override {
    process_calls_++;
    return {{"tag36h11", 7, 0, 42.0F, {}, 0.5, 0.5, {}}};
  }

  int calibration_calls() const { return calibration_calls_; }
  int process_calls() const { return process_calls_; }

 private:
  Calibration calibration_{};
  int calibration_calls_ = 0;
  int process_calls_ = 0;
};

struct ConcurrentState {
  std::mutex mutex;
  std::condition_variable condition;
  int entered = 0;
  bool release = false;
};

class BlockingBackend final : public DetectorBackend {
 public:
  explicit BlockingBackend(std::shared_ptr<ConcurrentState> state)
      : state_(std::move(state)) {}

  void SetCalibration(const Calibration&) override {}

  std::vector<Detection> Process(const GrayFrame&) override {
    std::unique_lock lock(state_->mutex);
    ++state_->entered;
    state_->condition.notify_all();
    state_->condition.wait(lock, [this] { return state_->release; });
    return {};
  }

 private:
  std::shared_ptr<ConcurrentState> state_;
};

void Release(const std::shared_ptr<ConcurrentState>& state) {
  {
    std::scoped_lock lock(state->mutex);
    state->release = true;
  }
  state->condition.notify_all();
}

struct LifecycleState {
  std::mutex mutex;
  std::condition_variable condition;
  bool process_entered = false;
  bool release_process = false;
  bool process_returned = false;
  bool destroyed = false;
  bool used_after_destroy = false;
};

class LifecycleBackend final : public DetectorBackend {
 public:
  explicit LifecycleBackend(std::shared_ptr<LifecycleState> state)
      : state_(std::move(state)) {}

  ~LifecycleBackend() override {
    std::scoped_lock lock(state_->mutex);
    state_->destroyed = true;
    state_->condition.notify_all();
  }

  void SetCalibration(const Calibration&) override {}

  std::vector<Detection> Process(const GrayFrame&) override {
    std::unique_lock lock(state_->mutex);
    state_->process_entered = true;
    state_->condition.notify_all();
    state_->condition.wait(lock, [this] { return state_->release_process; });
    state_->used_after_destroy = state_->destroyed;
    state_->process_returned = true;
    state_->condition.notify_all();
    return {};
  }

 private:
  std::shared_ptr<LifecycleState> state_;
};

void Release(const std::shared_ptr<LifecycleState>& state) {
  {
    std::scoped_lock lock(state->mutex);
    state->release_process = true;
  }
  state->condition.notify_all();
}

bool WaitForSize(DetectorRegistry& registry, std::size_t expected) {
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (registry.Size() == expected) {
      return true;
    }
    std::this_thread::sleep_for(1ms);
  }
  return registry.Size() == expected;
}

void TestCreateAndInvalidHandles() {
  RecordingBackend* first_backend = nullptr;
  RecordingBackend* second_backend = nullptr;
  int created = 0;
  DetectorRegistry registry([&](int, int, int) {
    auto backend = std::make_unique<RecordingBackend>();
    if (created++ == 0) {
      first_backend = backend.get();
    } else {
      second_backend = backend.get();
    }
    return backend;
  });

  const auto first = registry.Create(640, 480, 2);
  const auto second = registry.Create(640, 480, 2);
  Require(first != second, "Create returned duplicate handles");
  Require(registry.Size() == 2, "Create did not publish both handles");

  registry.SetCalibration(first, TestCalibration());
  Require(first_backend->calibration_calls() == 1,
          "SetCalibration did not reach the selected backend");
  const auto detections = registry.Process(second, Frame());
  Require(detections.size() == 1 && second_backend->process_calls() == 1,
          "Process did not reach the selected backend");

  constexpr std::int64_t kInvalidHandle = 999;
  RequireInvalidHandle(
      [&] { registry.SetCalibration(kInvalidHandle, TestCalibration()); },
      "SetCalibration");
  RequireInvalidHandle([&] { registry.Process(kInvalidHandle, Frame()); },
                       "Process");
  RequireInvalidHandle([&] { registry.Destroy(kInvalidHandle); }, "Destroy");

  registry.Destroy(first);
  registry.Destroy(second);
  Require(registry.Size() == 0, "Destroy did not remove all handles");
  RequireInvalidHandle([&] { registry.SetCalibration(first, TestCalibration()); },
                       "SetCalibration after Destroy");
  RequireInvalidHandle([&] { registry.Process(first, Frame()); },
                       "Process after Destroy");
  RequireInvalidHandle([&] { registry.Destroy(first); }, "Double Destroy");
  Require(registry.Size() == 0, "Double Destroy changed registry size");
}

void TestFactoryFailuresDoNotPublishSlots() {
  DetectorRegistry throwing_registry([](int, int, int) -> std::unique_ptr<DetectorBackend> {
    throw std::runtime_error("factory failed");
  });
  bool factory_threw = false;
  try {
    throwing_registry.Create(640, 480, 2);
  } catch (const std::runtime_error&) {
    factory_threw = true;
  }
  Require(factory_threw, "Create did not propagate factory exception");
  Require(throwing_registry.Size() == 0,
          "Throwing factory published a registry slot");

  DetectorRegistry null_registry(
      [](int, int, int) -> std::unique_ptr<DetectorBackend> { return nullptr; });
  bool null_factory_threw = false;
  try {
    null_registry.Create(640, 480, 2);
  } catch (const std::runtime_error&) {
    null_factory_threw = true;
  }
  Require(null_factory_threw, "Create accepted a null backend");
  Require(null_registry.Size() == 0, "Null factory published a registry slot");
}

void TestDifferentHandlesProcessConcurrently() {
  auto state = std::make_shared<ConcurrentState>();
  DetectorRegistry registry([state](int, int, int) {
    return std::make_unique<BlockingBackend>(state);
  });
  const auto first = registry.Create(640, 480, 2);
  const auto second = registry.Create(640, 480, 2);

  auto first_future = std::async(
      std::launch::async, [&] { registry.Process(first, Frame()); });
  std::future<void> second_future;
  try {
    second_future = std::async(
        std::launch::async, [&] { registry.Process(second, Frame()); });
  } catch (...) {
    const auto launch_error = std::current_exception();
    Release(state);
    std::exception_ptr worker_error;
    CaptureFuture(first_future, worker_error);
    std::rethrow_exception(launch_error);
  }

  bool both_entered = false;
  {
    std::unique_lock lock(state->mutex);
    both_entered = state->condition.wait_for(
        lock, kTimeout, [&] { return state->entered == 2; });
  }
  Release(state);

  std::exception_ptr worker_error;
  CaptureFuture(first_future, worker_error);
  CaptureFuture(second_future, worker_error);
  if (worker_error) {
    std::rethrow_exception(worker_error);
  }

  Require(both_entered,
          "Process serialized different handles behind the registry mutex");
}

void TestDestroyUnpublishesBeforeWaitingForProcess() {
  auto state = std::make_shared<LifecycleState>();
  DetectorRegistry registry([state](int, int, int) {
    return std::make_unique<LifecycleBackend>(state);
  });
  const auto handle = registry.Create(640, 480, 2);

  auto process_future = std::async(
      std::launch::async, [&] { registry.Process(handle, Frame()); });
  bool process_entered = false;
  {
    std::unique_lock lock(state->mutex);
    process_entered = state->condition.wait_for(
        lock, kTimeout, [&] { return state->process_entered; });
  }
  if (!process_entered) {
    Release(state);
    std::exception_ptr worker_error;
    CaptureFuture(process_future, worker_error);
    if (worker_error) {
      std::rethrow_exception(worker_error);
    }
    Require(false, "Process did not enter the backend");
  }

  std::future<void> destroy_future;
  try {
    destroy_future = std::async(
        std::launch::async, [&] { registry.Destroy(handle); });
  } catch (...) {
    const auto launch_error = std::current_exception();
    Release(state);
    std::exception_ptr worker_error;
    CaptureFuture(process_future, worker_error);
    std::rethrow_exception(launch_error);
  }

  std::future<bool> size_future;
  try {
    size_future = std::async(
        std::launch::async, [&] { return WaitForSize(registry, 0); });
  } catch (...) {
    const auto launch_error = std::current_exception();
    Release(state);
    std::exception_ptr worker_error;
    CaptureFuture(process_future, worker_error);
    CaptureFuture(destroy_future, worker_error);
    std::rethrow_exception(launch_error);
  }

  const bool size_probe_timed_out =
      size_future.wait_for(kTimeout) != std::future_status::ready;
  const bool destroy_completed_while_blocked =
      destroy_future.wait_for(0s) == std::future_status::ready;

  std::exception_ptr async_error;
  std::optional<bool> removed_before_wait;
  if (!size_probe_timed_out) {
    removed_before_wait = CaptureFuture(size_future, async_error);
  }

  std::future<bool> invalid_handle_future;
  bool invalid_handle_probe_timed_out = false;
  std::optional<bool> process_rejected;
  if (!async_error && removed_before_wait.value_or(false)) {
    try {
      invalid_handle_future = std::async(std::launch::async, [&] {
        return RejectsInvalidHandle(
            [&] { registry.Process(handle, Frame()); });
      });
    } catch (...) {
      async_error = std::current_exception();
    }
    if (invalid_handle_future.valid()) {
      invalid_handle_probe_timed_out =
          invalid_handle_future.wait_for(kTimeout) != std::future_status::ready;
      if (!invalid_handle_probe_timed_out) {
        process_rejected =
            CaptureFuture(invalid_handle_future, async_error);
      }
    }
  }

  Release(state);
  CaptureFuture(process_future, async_error);
  CaptureFuture(destroy_future, async_error);
  if (size_future.valid()) {
    removed_before_wait = CaptureFuture(size_future, async_error);
  }
  if (invalid_handle_future.valid()) {
    process_rejected = CaptureFuture(invalid_handle_future, async_error);
  }
  if (async_error) {
    std::rethrow_exception(async_error);
  }

  bool process_returned = false;
  bool destroyed = false;
  bool used_after_destroy = false;
  {
    std::scoped_lock lock(state->mutex);
    process_returned = state->process_returned;
    destroyed = state->destroyed;
    used_after_destroy = state->used_after_destroy;
  }

  Require(!size_probe_timed_out,
          "Size probe blocked behind in-flight Process");
  Require(removed_before_wait.value_or(false),
          "Destroy did not remove the handle before waiting for Process");
  Require(!destroy_completed_while_blocked,
          "Destroy completed while Process was blocked");
  Require(!invalid_handle_probe_timed_out,
          "Invalid-handle probe blocked after Destroy unpublished the handle");
  Require(process_rejected.value_or(false),
          "Process after concurrent Destroy accepted a handle");
  Require(process_returned, "Process did not return before Destroy");
  Require(destroyed, "Destroy did not reset the backend");
  Require(!used_after_destroy, "Process accessed the backend after it was reset");
}

}  // namespace

int main() {
  try {
    TestCreateAndInvalidHandles();
    TestFactoryFailuresDoNotPublishSlots();
    TestDestroyUnpublishesBeforeWaitingForProcess();
    TestDifferentHandlesProcessConcurrently();
  } catch (const std::exception& exception) {
    return (std::fprintf(stderr, "%s\n", exception.what()), 1);
  }
  return 0;
}
