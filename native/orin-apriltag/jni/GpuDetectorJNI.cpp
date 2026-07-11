#include <jni.h>

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "BuildInfo.h"
#include "CudaAprilTagDetector.h"
#include "DetectorRegistry.h"

namespace photon::cuda_apriltag {
namespace {

constexpr char kDetectionClassName[] =
    "edu/wpi/first/apriltag/AprilTagDetection";
constexpr char kDetectionConstructorDescriptor[] =
    "(Ljava/lang/String;IIF[DDD[D)V";

struct JniCache {
  std::mutex mutex;
  jclass detection_class = nullptr;
  jmethodID detection_constructor = nullptr;
};

JniCache& Cache() {
  static JniCache cache;
  return cache;
}

DetectorRegistry& Registry() {
  static DetectorRegistry registry([](int width, int height, int decimate) {
    return std::make_unique<CudaAprilTagDetector>(width, height, decimate);
  });
  return registry;
}

template <typename T>
class LocalRef {
 public:
  LocalRef(JNIEnv* env, T reference) : env_(env), reference_(reference) {}
  ~LocalRef() {
    if (reference_ != nullptr) {
      env_->DeleteLocalRef(reference_);
    }
  }

  LocalRef(const LocalRef&) = delete;
  LocalRef& operator=(const LocalRef&) = delete;

  T Get() const { return reference_; }
  T Release() { return std::exchange(reference_, nullptr); }

 private:
  JNIEnv* env_;
  T reference_;
};

void ThrowRuntimeException(JNIEnv* env, const char* message) noexcept {
  if (env->ExceptionCheck()) {
    return;
  }
  jclass exception_class = env->FindClass("java/lang/RuntimeException");
  if (exception_class == nullptr) {
    return;
  }
  env->ThrowNew(exception_class, message);
  env->DeleteLocalRef(exception_class);
}

template <typename Result, typename Callable>
Result GuardJni(JNIEnv* env, Result failure, Callable&& callable) noexcept {
  try {
    return std::forward<Callable>(callable)();
  } catch (const std::exception& error) {
    ThrowRuntimeException(env, error.what());
  } catch (...) {
    ThrowRuntimeException(env, "Unknown CUDA detector failure");
  }
  return failure;
}

template <typename Callable>
void GuardJniVoid(JNIEnv* env, Callable&& callable) noexcept {
  try {
    std::forward<Callable>(callable)();
  } catch (const std::exception& error) {
    ThrowRuntimeException(env, error.what());
  } catch (...) {
    ThrowRuntimeException(env, "Unknown CUDA detector failure");
  }
}

std::int64_t CheckedHandle(jlong handle) {
  if (handle <= 0) {
    throw std::invalid_argument("Invalid CUDA detector handle");
  }
  return static_cast<std::int64_t>(handle);
}

const std::uint8_t* CheckedAddress(jlong address, std::size_t span_bytes) {
  if (address <= 0) {
    throw std::invalid_argument("Grayscale image address must be nonzero");
  }
  const auto value = static_cast<std::uint64_t>(address);
  if (value > std::numeric_limits<std::uintptr_t>::max()) {
    throw std::overflow_error("Grayscale image address is out of range");
  }
  const auto base = static_cast<std::uintptr_t>(value);
  const std::size_t last_offset = span_bytes - 1;
  if (last_offset > std::numeric_limits<std::uintptr_t>::max() ||
      base > std::numeric_limits<std::uintptr_t>::max() -
                 static_cast<std::uintptr_t>(last_offset)) {
    throw std::overflow_error(
        "Grayscale image address range exceeds UINTPTR_MAX");
  }
  return reinterpret_cast<const std::uint8_t*>(base);
}

std::size_t CheckedStride(jlong stride, jint width) {
  if (stride <= 0) {
    throw std::invalid_argument("Grayscale image stride must be positive");
  }
  const auto value = static_cast<std::uint64_t>(stride);
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("Grayscale image stride is out of range");
  }
  if (width <= 0 || value < static_cast<std::uint64_t>(width)) {
    throw std::invalid_argument(
        "Grayscale image width must be positive and no greater than stride");
  }
  return static_cast<std::size_t>(value);
}

std::size_t CheckedFrameSpan(jint width, jint height,
                             std::size_t stride_bytes) {
  if (width <= 0) {
    throw std::invalid_argument("Grayscale image width must be positive");
  }
  if (height <= 0) {
    throw std::invalid_argument("Grayscale image height must be positive");
  }
  const auto row_bytes = static_cast<std::size_t>(width);
  const auto rows_before_last = static_cast<std::size_t>(height - 1);
  if (rows_before_last >
      (std::numeric_limits<std::size_t>::max() - row_bytes) / stride_bytes) {
    throw std::overflow_error("Grayscale image span overflows size_t");
  }
  const std::size_t span_bytes =
      rows_before_last * stride_bytes + row_bytes;
  if (span_bytes > static_cast<std::size_t>(
                       std::numeric_limits<std::ptrdiff_t>::max())) {
    throw std::overflow_error("Grayscale image span exceeds PTRDIFF_MAX");
  }
  return span_bytes;
}

jlong CheckedJlong(std::size_t value) {
  if (value > static_cast<std::size_t>(std::numeric_limits<jlong>::max())) {
    throw std::overflow_error("CUDA memory size exceeds Java long range");
  }
  return static_cast<jlong>(value);
}

struct DetectionClassSnapshot {
  jclass local_class;
  jmethodID constructor;
};

DetectionClassSnapshot SnapshotDetectionClass(JNIEnv* env) {
  auto& cache = Cache();
  std::scoped_lock lock(cache.mutex);
  if (cache.detection_class == nullptr ||
      cache.detection_constructor == nullptr) {
    throw std::runtime_error("AprilTagDetection JNI cache is unavailable");
  }
  auto local_class =
      static_cast<jclass>(env->NewLocalRef(cache.detection_class));
  if (local_class == nullptr) {
    if (env->ExceptionCheck()) {
      throw std::runtime_error("Unable to retain AprilTagDetection class");
    }
    throw std::runtime_error("Unable to create AprilTagDetection class ref");
  }
  return {local_class, cache.detection_constructor};
}

jdoubleArray NewDoubleArray(JNIEnv* env, const double* values,
                            jsize length) {
  jdoubleArray array = env->NewDoubleArray(length);
  if (array == nullptr) {
    return nullptr;
  }
  env->SetDoubleArrayRegion(array, 0, length, values);
  if (env->ExceptionCheck()) {
    env->DeleteLocalRef(array);
    return nullptr;
  }
  return array;
}

jobjectArray BuildDetectionArray(JNIEnv* env,
                                 const std::vector<Detection>& detections) {
  const DetectionClassSnapshot snapshot = SnapshotDetectionClass(env);
  LocalRef<jclass> detection_class(env, snapshot.local_class);
  if (detections.size() >
      static_cast<std::size_t>(std::numeric_limits<jsize>::max())) {
    throw std::overflow_error("Too many AprilTag detections for Java array");
  }

  LocalRef<jobjectArray> result(
      env, env->NewObjectArray(static_cast<jsize>(detections.size()),
                               detection_class.Get(), nullptr));
  if (result.Get() == nullptr) {
    return nullptr;
  }

  for (std::size_t index = 0; index < detections.size(); ++index) {
    const Detection& detection = detections[index];
    LocalRef<jstring> family(env, env->NewStringUTF(detection.family.c_str()));
    if (family.Get() == nullptr) {
      return nullptr;
    }
    LocalRef<jdoubleArray> homography(
        env, NewDoubleArray(env, detection.homography.data(),
                            static_cast<jsize>(detection.homography.size())));
    if (homography.Get() == nullptr) {
      return nullptr;
    }
    LocalRef<jdoubleArray> corners(
        env, NewDoubleArray(env, detection.corners.data(),
                            static_cast<jsize>(detection.corners.size())));
    if (corners.Get() == nullptr) {
      return nullptr;
    }
    LocalRef<jobject> java_detection(
        env, env->NewObject(detection_class.Get(), snapshot.constructor,
                            family.Get(), static_cast<jint>(detection.id),
                            static_cast<jint>(detection.hamming),
                            static_cast<jfloat>(detection.decision_margin),
                            homography.Get(),
                            static_cast<jdouble>(detection.center_x),
                            static_cast<jdouble>(detection.center_y),
                            corners.Get()));
    if (java_detection.Get() == nullptr) {
      return nullptr;
    }
    env->SetObjectArrayElement(result.Get(), static_cast<jsize>(index),
                               java_detection.Get());
    if (env->ExceptionCheck()) {
      return nullptr;
    }
  }
  return result.Release();
}

}  // namespace
}  // namespace photon::cuda_apriltag

using photon::cuda_apriltag::BuildDetectionArray;
using photon::cuda_apriltag::Cache;
using photon::cuda_apriltag::Calibration;
using photon::cuda_apriltag::CheckedAddress;
using photon::cuda_apriltag::CheckedFrameSpan;
using photon::cuda_apriltag::CheckedHandle;
using photon::cuda_apriltag::CheckedJlong;
using photon::cuda_apriltag::CheckedStride;
using photon::cuda_apriltag::GrayFrame;
using photon::cuda_apriltag::GuardJni;
using photon::cuda_apriltag::GuardJniVoid;
using photon::cuda_apriltag::Registry;

extern "C" {

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  JNIEnv* env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_8) != JNI_OK ||
      env == nullptr) {
    return JNI_ERR;
  }

  jclass local_class = env->FindClass(
      photon::cuda_apriltag::kDetectionClassName);
  if (local_class == nullptr) {
    return JNI_ERR;
  }
  jmethodID constructor = env->GetMethodID(
      local_class, "<init>",
      photon::cuda_apriltag::kDetectionConstructorDescriptor);
  if (constructor == nullptr) {
    env->DeleteLocalRef(local_class);
    return JNI_ERR;
  }
  jclass global_class =
      static_cast<jclass>(env->NewGlobalRef(local_class));
  env->DeleteLocalRef(local_class);
  if (global_class == nullptr) {
    return JNI_ERR;
  }

  auto& cache = Cache();
  std::scoped_lock lock(cache.mutex);
  cache.detection_class = global_class;
  cache.detection_constructor = constructor;
  return JNI_VERSION_1_8;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void*) {
  JNIEnv* env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_8) != JNI_OK ||
      env == nullptr) {
    return;
  }
  auto& cache = Cache();
  std::scoped_lock lock(cache.mutex);
  if (cache.detection_class != nullptr) {
    env->DeleteGlobalRef(cache.detection_class);
    cache.detection_class = nullptr;
  }
  cache.detection_constructor = nullptr;
}

JNIEXPORT jlong JNICALL
Java_org_photonvision_jni_GpuDetectorJNI_createGpuDetector(
    JNIEnv* env, jclass, jint width, jint height, jint decimate) {
  return GuardJni<jlong>(env, -1, [&] {
    return static_cast<jlong>(Registry().Create(width, height, decimate));
  });
}

JNIEXPORT void JNICALL Java_org_photonvision_jni_GpuDetectorJNI_setCalibration(
    JNIEnv* env, jclass, jlong handle, jdouble fx, jdouble fy, jdouble cx,
    jdouble cy, jdouble k1, jdouble k2, jdouble p1, jdouble p2, jdouble k3) {
  GuardJniVoid(env, [&] {
    Registry().SetCalibration(CheckedHandle(handle),
                              Calibration{fx, fy, cx, cy, k1, k2, p1, p2, k3});
  });
}

JNIEXPORT jobjectArray JNICALL
Java_org_photonvision_jni_GpuDetectorJNI_processGray(
    JNIEnv* env, jclass, jlong handle, jlong address, jint width, jint height,
    jlong stride) {
  return GuardJni<jobjectArray>(env, nullptr, [&] {
    const std::int64_t checked_handle = CheckedHandle(handle);
    const std::size_t checked_stride = CheckedStride(stride, width);
    const std::size_t span_bytes =
        CheckedFrameSpan(width, height, checked_stride);
    const GrayFrame frame{CheckedAddress(address, span_bytes), width, height,
                          checked_stride};
    return BuildDetectionArray(env, Registry().Process(checked_handle, frame));
  });
}

JNIEXPORT void JNICALL Java_org_photonvision_jni_GpuDetectorJNI_destroyGpuDetector(
    JNIEnv* env, jclass, jlong handle) {
  GuardJniVoid(env, [&] { Registry().Destroy(CheckedHandle(handle)); });
}

JNIEXPORT jlong JNICALL
Java_org_photonvision_jni_GpuDetectorJNI_getCudaFreeMemoryBytes(JNIEnv* env,
                                                                jclass) {
  return GuardJni<jlong>(env, 0, [&] {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMemGetInfo failed: ") +
                               cudaGetErrorString(status));
    }
    return CheckedJlong(free_bytes);
  });
}

JNIEXPORT jlong JNICALL
Java_org_photonvision_jni_GpuDetectorJNI_getCudaTotalMemoryBytes(JNIEnv* env,
                                                                 jclass) {
  return GuardJni<jlong>(env, 0, [&] {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMemGetInfo failed: ") +
                               cudaGetErrorString(status));
    }
    return CheckedJlong(total_bytes);
  });
}

JNIEXPORT jstring JNICALL
Java_org_photonvision_jni_GpuDetectorJNI_getBuildInfo(JNIEnv* env, jclass) {
  return GuardJni<jstring>(env, nullptr, [&] {
    std::ostringstream info;
    info << "sourceRevision=" << PV_CUDA_SOURCE_REVISION
         << ", l4t=" << PV_CUDA_L4T_RELEASE
         << ", architecture=sm_" << PV_CUDA_ARCHITECTURE
         << ", cudaCompiler=" << PV_CUDA_COMPILER
         << ", buildTimestamp=" << PV_CUDA_BUILD_TIMESTAMP;
    return env->NewStringUTF(info.str().c_str());
  });
}

}  // extern "C"
