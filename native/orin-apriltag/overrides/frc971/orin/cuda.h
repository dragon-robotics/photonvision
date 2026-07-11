#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// ABSL replace
#define FATAL true
#define INFO true
#define LOG(...) std::cout
#define VLOG(...) std::cout

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace frc971::apriltag {

inline void CheckCuda(cudaError_t status, const char* expression,
                      const char* file, int line) {
  if (status == cudaSuccess) return;
  throw std::runtime_error(std::string("CUDA failure: ") + expression +
                           " at " + file + ":" + std::to_string(line) +
                           ": " + cudaGetErrorString(status));
}

inline void CheckCudaNoThrow(cudaError_t status, const char* expression,
                             const char* file, int line) noexcept {
  if (status == cudaSuccess) return;
  const char* error = cudaGetErrorString(status);
  std::fprintf(stderr, "CUDA cleanup failure: %s at %s:%d: %s\n", expression,
               file, line, error == nullptr ? "unknown CUDA error" : error);
}

class CudaCheckContinuation {
 public:
  template <typename T>
  const CudaCheckContinuation& operator<<(const T&) const noexcept {
    return *this;
  }
};

inline CudaCheckContinuation CheckedCuda(cudaError_t status,
                                         const char* expression,
                                         const char* file, int line) {
  CheckCuda(status, expression, file, line);
  return {};
}

template <typename T>
std::size_t CheckedAllocationBytes(std::size_t size) {
  if (size > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::overflow_error("CUDA allocation size overflows size_t");
  }
  return size * sizeof(T);
}

// cuda.cc appends a diagnostic message to this macro. The continuation keeps
// that pinned call shape while CheckCuda provides the throwing behavior.
#define CHECK_CUDA(expression)                                           \
  ::frc971::apriltag::CheckedCuda((expression), #expression, __FILE__,  \
                                  __LINE__)
#define CHECK(condition)                                                    \
  do {                                                                      \
    if (!(condition)) throw std::runtime_error("Check failed: " #condition); \
  } while (false)
#define CHECK_EQ(left, right) CHECK((left) == (right));
#define CHECK_LE(left, right) CHECK((left) <= (right));
#define CHECK_LT(left, right) CHECK((left) < (right));

template <typename InputType, typename OutputType, typename ConversionOp,
          typename OffsetT>
class TransformOutputIterator;

template <typename Iterator>
class TransformOutputOffset {
 public:
  __host__ __device__ TransformOutputOffset(Iterator iterator,
                                             std::int64_t offset)
      : iterator_(iterator), offset_(offset) {}

  template <typename Distance>
  __host__ __device__ TransformOutputOffset operator+(
      Distance distance) const {
    return TransformOutputOffset(iterator_,
                                 offset_ + static_cast<std::int64_t>(distance));
  }

  __host__ __device__ auto operator*() const { return iterator_[offset_]; }

 private:
  Iterator iterator_;
  std::int64_t offset_;
};

template <typename InputType, typename OutputType, typename ConversionOp,
          typename OffsetT, typename Distance>
__host__ __device__ auto operator+(
    TransformOutputIterator<InputType, OutputType, ConversionOp, OffsetT>
        iterator,
    Distance distance) {
  using Iterator =
      TransformOutputIterator<InputType, OutputType, ConversionOp, OffsetT>;
  return TransformOutputOffset<Iterator>(iterator,
                                         static_cast<std::int64_t>(distance));
}

// Class to manage the lifetime of a Cuda stream.  This is used to provide
// relative ordering between kernels on the same stream.
class CudaStream {
 public:
  CudaStream() { CHECK_CUDA(cudaStreamCreate(&stream_)); }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  virtual ~CudaStream() noexcept {
    if (stream_ != nullptr) {
      CheckCudaNoThrow(cudaStreamDestroy(stream_), "cudaStreamDestroy(stream_)",
                       __FILE__, __LINE__);
    }
  }

  cudaStream_t get() { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

// Class to manage the lifetime of a Cuda Event.  Cuda events are used for
// timing events on a stream.
class CudaEvent {
 public:
  CudaEvent() { CHECK_CUDA(cudaEventCreate(&event_)); }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  virtual ~CudaEvent() noexcept {
    if (event_ != nullptr) {
      CheckCudaNoThrow(cudaEventDestroy(event_), "cudaEventDestroy(event_)",
                       __FILE__, __LINE__);
    }
  }

  void Record(CudaStream* stream) {
    CHECK_CUDA(cudaEventRecord(event_, stream->get()));
  }

  std::chrono::nanoseconds ElapsedTime(const CudaEvent& start) {
    float ms = 0.0F;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start.event_, event_));
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<float, std::milli>(ms));
  }

  void Synchronize() { CHECK_CUDA(cudaEventSynchronize(event_)); }

 private:
  cudaEvent_t event_ = nullptr;
};

// Class to manage the lifetime of page locked host memory for fast copies back
// to host memory.
template <typename T>
class HostMemory {
 public:
  explicit HostMemory(std::size_t size) : size_(size) {
    CHECK_CUDA(cudaMallocHost(reinterpret_cast<void**>(&memory_),
                              CheckedAllocationBytes<T>(size_)));
  }
  HostMemory(const HostMemory&) = delete;
  HostMemory& operator=(const HostMemory&) = delete;

  virtual ~HostMemory() noexcept {
    if (memory_ != nullptr) {
      CheckCudaNoThrow(cudaFreeHost(memory_), "cudaFreeHost(memory_)", __FILE__,
                       __LINE__);
    }
  }

  T* get() { return memory_; }
  const T* get() const { return memory_; }
  std::size_t size() const { return size_; }

  void MemcpyFrom(const T* other) {
    std::memcpy(memory_, other, CheckedAllocationBytes<T>(size_));
  }
  void MemcpyTo(T* other) {
    std::memcpy(other, memory_, CheckedAllocationBytes<T>(size_));
  }

 private:
  T* memory_ = nullptr;
  std::size_t size_ = 0;
};

// Class to manage the lifetime of device memory.
template <typename T>
class GpuMemory {
 public:
  explicit GpuMemory(std::size_t size) : size_(size) {
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&memory_),
                          CheckedAllocationBytes<T>(size_)));
  }
  GpuMemory(const GpuMemory&) = delete;
  GpuMemory& operator=(const GpuMemory&) = delete;

  virtual ~GpuMemory() noexcept {
    if (memory_ != nullptr) {
      CheckCudaNoThrow(cudaFree(memory_), "cudaFree(memory_)", __FILE__,
                       __LINE__);
    }
  }

  T* get() { return memory_; }
  const T* get() const { return memory_; }
  std::size_t size() const { return size_; }

  void MemcpyAsyncFrom(const T* host_memory, CudaStream* stream) {
    CHECK_CUDA(cudaMemcpyAsync(memory_, host_memory,
                               CheckedAllocationBytes<T>(size_),
                               cudaMemcpyHostToDevice, stream->get()));
  }
  void MemcpyAsyncFrom(const HostMemory<T>* host_memory, CudaStream* stream) {
    MemcpyAsyncFrom(host_memory->get(), stream);
  }

  void MemcpyAsyncTo(T* host_memory, std::size_t size,
                     CudaStream* stream) const {
    CHECK_LE(size, size_);
    CHECK_CUDA(cudaMemcpyAsync(host_memory, memory_,
                               CheckedAllocationBytes<T>(size),
                               cudaMemcpyDeviceToHost, stream->get()));
  }
  void MemcpyAsyncTo(T* host_memory, CudaStream* stream) const {
    MemcpyAsyncTo(host_memory, size_, stream);
  }
  void MemcpyAsyncTo(HostMemory<T>* host_memory, CudaStream* stream) const {
    MemcpyAsyncTo(host_memory->get(), stream);
  }

  void MemcpyFrom(const T* host_memory) {
    CHECK_CUDA(cudaMemcpy(memory_, host_memory,
                          CheckedAllocationBytes<T>(size_),
                          cudaMemcpyHostToDevice));
  }
  void MemcpyFrom(const HostMemory<T>* host_memory) {
    MemcpyFrom(host_memory->get());
  }

  void MemcpyTo(T* host_memory, std::size_t size) const {
    CHECK_LE(size, size_);
    CHECK_CUDA(cudaMemcpy(host_memory, memory_,
                          CheckedAllocationBytes<T>(size),
                          cudaMemcpyDeviceToHost));
  }
  void MemcpyTo(T* host_memory) const { MemcpyTo(host_memory, size_); }
  void MemcpyTo(HostMemory<T>* host_memory) const {
    MemcpyTo(host_memory->get());
  }

  void MemsetAsync(const std::uint8_t value, CudaStream* stream) const {
    CHECK_CUDA(cudaMemsetAsync(memory_, value,
                               CheckedAllocationBytes<T>(size_),
                               stream->get()));
  }

  std::vector<T> Copy(std::size_t size) const {
    CHECK_LE(size, size_);
    std::vector<T> result(size);
    MemcpyTo(result.data(), size);
    return result;
  }

  std::vector<T> Copy() const { return Copy(size_); }

 private:
  T* memory_ = nullptr;
  const std::size_t size_;
};

void CheckAndSynchronize(std::string_view message = "");
void MaybeCheckAndSynchronize();
void MaybeCheckAndSynchronize(std::string_view message);

}  // namespace frc971::apriltag
