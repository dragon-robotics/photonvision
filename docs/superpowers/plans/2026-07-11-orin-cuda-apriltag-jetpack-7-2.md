# JetPack 7.2 CUDA AprilTag Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a memory-safe, dual-camera CUDA AprilTag detector for the Jetson Orin Nano on JetPack 7.2.

**Architecture:** Vendor the pinned FRC Team 4143/FRC 971 CUDA source, replace its WPILib/OpenCV-dependent JNI wrapper with a raw-JNI RAII implementation, and build the native library directly on the Jetson. The existing AprilTag pipeline retains CUDA, TensorRT ROI, and CPU selection, while explicit detector lifecycle and systemd limits prevent a native failure from exhausting the board.

**Tech Stack:** Java 17, JUnit 5, C++20, CUDA 13.2, CMake 3.28+, Ninja, raw JNI, PhotonVision 2026.3.4, JetPack 7.2 / L4T R39.2, PowerShell, Bash, systemd

## Global Constraints

- Work on branch `image/orin-nano` and preserve unrelated dirty files.
- Target Jetson Orin Nano 8 GB, AArch64, JetPack 7.2 / L4T R39.2, compute architecture 8.7.
- Keep Java 17 and PhotonVision 2026.3.4 compatibility.
- Build CUDA code on the Jetson; do not cross-compile CUDA on Windows.
- Do not depend on a separately built allwpilib tree, `libwpiutil.so`, or a C++ OpenCV ABI.
- Keep CUDA disabled by default until all acceptance gates pass.
- Preserve detector order: CUDA, TensorRT ROI when enabled, then CPU AprilTag.
- A valid empty CUDA detection result must not invoke a fallback.
- A CUDA exception or unavailable backend must invoke the configured fallback.
- Do not alter USB quirks, camera matching, exposure behavior, TensorRT model behavior, CSI support, or Isaac ROS support.
- Follow red-green-refactor for every Java and native behavior change.
- Never run the two-camera CUDA test without the memory monitor and `MemoryMax=6G` active.

## File Map

- `native/orin-apriltag/UPSTREAM.md`: pinned source revision and local modifications.
- `native/orin-apriltag/LICENSE-4143.txt`: upstream Apache-2.0 license.
- `native/orin-apriltag/VENDOR_MANIFEST.sha256`: SHA-256 manifest for every vendored upstream file.
- `native/orin-apriltag/upstream/frc971/orin/`: pinned CUDA implementation.
- `native/orin-apriltag/upstream/third_party/apriltag/`: pinned AprilTag C implementation and license.
- `native/orin-apriltag/overrides/frc971/orin/cuda.h`: local throwing-check override staged at build time without modifying pinned source.
- `native/orin-apriltag/jni/DetectorRegistry.h`: handle registry and testable detector interface.
- `native/orin-apriltag/jni/DetectorRegistry.cpp`: concurrent handle lifecycle.
- `native/orin-apriltag/jni/CudaAprilTagDetector.h`: RAII detector declaration.
- `native/orin-apriltag/jni/CudaAprilTagDetector.cu`: raw grayscale processing and detection value conversion.
- `native/orin-apriltag/jni/GpuDetectorJNI.cpp`: raw JNI entry points and exception translation.
- `native/orin-apriltag/tests/DetectorRegistryTest.cpp`: host-only lifecycle/concurrency tests.
- `native/orin-apriltag/tests/CudaDetectorSmokeTest.cpp`: Jetson CUDA create/process/destroy test.
- `native/orin-apriltag/CMakeLists.txt`: native shared library and CTest targets.
- `native/orin-apriltag/BuildInfo.h.in`: native build metadata template.
- `scripts/build-orin-cuda-apriltag.sh`: target-side prerequisite check and build.
- `scripts/monitor-orin-photonvision.sh`: bounded-memory acceptance monitor.
- `scripts/tests/verify-orin-cuda-vendor.ps1`: vendored-source provenance test.
- `scripts/tests/monitor-orin-photonvision-test.sh`: monitor threshold tests.
- `scripts/orin-nvme-team-image.sh`: native install, service limits, and health check.
- `photon-core/src/main/java/org/photonvision/jni/GpuDetectorJNI.java`: JNI facade.
- `photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagDetectionCudaPipe.java`: Java detector lifecycle and frame bridge.
- `photon-core/src/main/java/org/photonvision/vision/pipeline/AprilTagPipeline.java`: detector enable/disable and fallback integration.
- `photon-core/src/test/java/org/photonvision/vision/pipe/impl/AprilTagDetectionCudaPipeTest.java`: Java pipe tests.
- `photon-core/src/test/java/org/photonvision/vision/pipeline/AprilTagPipelineMLBehaviorTest.java`: detector-selection tests.
- `docs/orin-nano-nvme-team-image.md`: build, deployment, and measured results.

---

### Task 1: Vendor the Pinned CUDA Detector Source

**Files:**
- Create: `scripts/tests/verify-orin-cuda-vendor.ps1`
- Create: `native/orin-apriltag/UPSTREAM.md`
- Create: `native/orin-apriltag/LICENSE-4143.txt`
- Create: `native/orin-apriltag/VENDOR_MANIFEST.sha256`
- Create: `native/orin-apriltag/upstream/frc971/orin/**`
- Create: `native/orin-apriltag/upstream/third_party/apriltag/**`

**Interfaces:**
- Consumes: upstream repository `https://github.com/FRC-Team-4143/GpuDetectorJNI.git` at commit `ef9fc1ec7e43116849e71fef1ab335ba630274a7`.
- Produces: a self-contained, licensed native source tree used by every later native task.

- [ ] **Step 1: Write the failing vendor-layout test**

Create `scripts/tests/verify-orin-cuda-vendor.ps1`:

```powershell
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "../..")
$required = @(
    "native/orin-apriltag/UPSTREAM.md",
    "native/orin-apriltag/LICENSE-4143.txt",
    "native/orin-apriltag/upstream/frc971/orin/971apriltag.h",
    "native/orin-apriltag/upstream/frc971/orin/971apriltag.cu",
    "native/orin-apriltag/upstream/third_party/apriltag/LICENSE.md",
    "native/orin-apriltag/upstream/third_party/apriltag/apriltag.h"
)

$missing = $required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $root $_)) }
if ($missing.Count -ne 0) {
    throw "Missing vendored CUDA files: $($missing -join ', ')"
}

$provenance = Get-Content -LiteralPath (Join-Path $root "native/orin-apriltag/UPSTREAM.md") -Raw
if (-not $provenance.Contains("ef9fc1ec7e43116849e71fef1ab335ba630274a7")) {
    throw "UPSTREAM.md does not pin the approved GpuDetectorJNI revision"
}
```

- [ ] **Step 2: Run the test and verify the source is absent**

Run:

```powershell
powershell -NoProfile -File scripts/tests/verify-orin-cuda-vendor.ps1
```

Expected: FAIL with `Missing vendored CUDA files`.

- [ ] **Step 3: Copy the exact upstream source mechanically**

Run from the repository root:

```powershell
$reference = Join-Path $env:TEMP "gpudetectorjni-ef9fc1e"
git clone https://github.com/FRC-Team-4143/GpuDetectorJNI.git $reference
git -C $reference checkout ef9fc1ec7e43116849e71fef1ab335ba630274a7
New-Item -ItemType Directory -Force native/orin-apriltag/upstream | Out-Null
Copy-Item -Recurse -Force "$reference/frc971" native/orin-apriltag/upstream/
Copy-Item -Recurse -Force "$reference/third_party" native/orin-apriltag/upstream/
Copy-Item -Force "$reference/LICENSE.txt" native/orin-apriltag/LICENSE-4143.txt
```

Do not copy the upstream `GpuDetectorJNI.cc`, `CMakeLists.txt`, build outputs, or Git metadata. The new JNI adapter replaces them.

- [ ] **Step 4: Add exact provenance**

Create `native/orin-apriltag/UPSTREAM.md` with:

```markdown
# Upstream Provenance

- Repository: https://github.com/FRC-Team-4143/GpuDetectorJNI
- Revision: ef9fc1ec7e43116849e71fef1ab335ba630274a7
- Imported: 2026-07-11
- Upstream target: JetPack 6.2
- Local target: JetPack 7.2 / L4T R39.2

The `frc971/orin` and `third_party/apriltag` trees are pinned upstream source.
PhotonVision supplies a replacement raw-JNI adapter, RAII handle registry,
JetPack 7.2 build, error handling, tests, and deployment scripts.
```

- [ ] **Step 5: Verify the vendor tree**

Run:

```powershell
powershell -NoProfile -File scripts/tests/verify-orin-cuda-vendor.ps1
git diff --check -- native/orin-apriltag/UPSTREAM.md native/orin-apriltag/LICENSE-4143.txt native/orin-apriltag/VENDOR_MANIFEST.sha256 scripts/tests/verify-orin-cuda-vendor.ps1
```

Expected: the verifier requires exact manifest path-set equality and SHA-256 matches, while the diff check succeeds for locally authored files. Vendored bytes are integrity-checked and intentionally excluded from whitespace normalization.

- [ ] **Step 6: Commit the pinned source**

```powershell
git add native/orin-apriltag scripts/tests/verify-orin-cuda-vendor.ps1
git commit -m "build: vendor Orin CUDA AprilTag source"
```

---

### Task 2: Add the JetPack 7.2 Build Scaffold

**Files:**
- Create: `native/orin-apriltag/CMakeLists.txt`
- Create: `native/orin-apriltag/BuildInfo.h.in`
- Create: `scripts/build-orin-cuda-apriltag.sh`

**Interfaces:**
- Consumes: Task 1 vendored source, JDK 17 headers, CUDA 13.2, CMake 3.28+, Ninja.
- Produces: a verified target-side build script, generated build metadata, and the static `apriltag_vendor` build consumed by Tasks 3 and 4.

- [ ] **Step 1: Add a prerequisite-only build-script mode**

Create `scripts/build-orin-cuda-apriltag.sh` with these arguments and checks:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build/orin-apriltag"
CHECK_ONLY="no"
SOURCE_REVISION="${PV_SOURCE_REVISION:-}"
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir) BUILD_DIR="${2:?missing build directory}"; shift 2 ;;
        --source-revision) SOURCE_REVISION="${2:?missing source revision}"; shift 2 ;;
        --check-only) CHECK_ONLY="yes"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ "$(uname -m)" == "aarch64" ]] || { echo "CUDA build requires AArch64" >&2; exit 1; }
grep -q '^# R39 (release), REVISION: 2.0' /etc/nv_tegra_release || {
    echo "CUDA build requires JetPack 7.2 / L4T R39.2" >&2
    exit 1
}
command -v nvcc >/dev/null || { echo "Install nvidia-cuda-dev" >&2; exit 1; }
command -v cmake >/dev/null || { echo "Install cmake" >&2; exit 1; }
command -v ninja >/dev/null || { echo "Install ninja-build" >&2; exit 1; }
command -v javac >/dev/null || { echo "Install openjdk-17-jdk" >&2; exit 1; }
[[ "$(javac -version 2>&1)" == javac\ 17.* ]] || { echo "JDK 17 is required" >&2; exit 1; }

[[ "${CHECK_ONLY}" == "yes" ]] && exit 0

if [[ -z "${SOURCE_REVISION}" ]]; then
    SOURCE_REVISION="$(git -C "${ROOT}" rev-parse --short=12 HEAD 2>/dev/null || true)"
fi
[[ -n "${SOURCE_REVISION}" ]] || {
    echo "Pass --source-revision when building from a copied source tree" >&2
    exit 1
}

cmake -S "${ROOT}/native/orin-apriltag" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DPV_SOURCE_REVISION="${SOURCE_REVISION}"
cmake --build "${BUILD_DIR}" --parallel "$(nproc)"
ctest --test-dir "${BUILD_DIR}" --output-on-failure
```

- [ ] **Step 2: Prove the current Jetson lacks the build prerequisites**

Copy only the script to the Jetson and run:

```powershell
scp scripts/build-orin-cuda-apriltag.sh doug@192.168.50.97:/tmp/
ssh doug@192.168.50.97 "bash /tmp/build-orin-cuda-apriltag.sh --check-only"
```

Expected: FAIL with `Install nvidia-cuda-dev`. The 2026-07-11 baseline has Java runtime 17.0.19 but no `nvcc`, CMake, Ninja, or `javac`.

- [ ] **Step 3: Install the exact target toolchain**

Run on the Jetson:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-cuda-dev openjdk-17-jdk cmake ninja-build build-essential
```

Expected package sources: NVIDIA JetPack 7.2 `nvidia-cuda-dev` candidate `7.2-b187`, CUDA compiler 13.2, Ubuntu 24.04 CMake 3.28.3, Ninja 1.11.1, and OpenJDK 17.0.19.

- [ ] **Step 4: Verify the prerequisite mode passes**

```bash
bash /tmp/build-orin-cuda-apriltag.sh --check-only
nvcc --version
cmake --version
ninja --version
javac -version
```

Expected: check succeeds; `nvcc` reports CUDA 13.2 and `javac` reports 17.

- [ ] **Step 5: Add the CMake target and build metadata template**

Create `native/orin-apriltag/BuildInfo.h.in`:

```cpp
#pragma once
#define PV_CUDA_SOURCE_REVISION "@PV_SOURCE_REVISION@"
#define PV_CUDA_L4T_RELEASE "R39.2"
#define PV_CUDA_ARCHITECTURE "87"
#define PV_CUDA_COMPILER "@CMAKE_CUDA_COMPILER_VERSION@"
#define PV_CUDA_BUILD_TIMESTAMP "@PV_BUILD_TIMESTAMP@"
```

Create `native/orin-apriltag/CMakeLists.txt` with `project(PhotonOrinAprilTag LANGUAGES C CXX CUDA)`, C++20/CUDA20, `find_package(JNI REQUIRED)`, `find_package(CUDAToolkit REQUIRED)`, `CMAKE_CUDA_ARCHITECTURES=87`, `enable_testing()`, generated `BuildInfo.h`, and a static `apriltag_vendor` target from the vendored C source. Set `PV_BUILD_TIMESTAMP` with `string(TIMESTAMP PV_BUILD_TIMESTAMP UTC)`. Do not declare the shared JNI target until Task 4 supplies all of its source files.

The static target must expose `upstream/third_party/apriltag` headers and compile with position-independent code so Task 4 can link it into `lib971apriltag.so`.

- [ ] **Step 6: Verify the vendored AprilTag target builds**

Copy `native/orin-apriltag` to the Jetson and run:

```bash
cmake -S native/orin-apriltag -B build/orin-apriltag -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build/orin-apriltag --target apriltag_vendor
```

Expected: CMake configuration and the static vendored AprilTag build succeed without WPILib or OpenCV.

- [ ] **Step 7: Commit the build scaffold**

```powershell
git add native/orin-apriltag/CMakeLists.txt native/orin-apriltag/BuildInfo.h.in scripts/build-orin-cuda-apriltag.sh
git commit -m "build: add JetPack 7.2 CUDA build scaffold"
```

---

### Task 3: Implement the Thread-Safe Native Handle Registry

**Files:**
- Create: `native/orin-apriltag/jni/DetectorRegistry.h`
- Create: `native/orin-apriltag/jni/DetectorRegistry.cpp`
- Create: `native/orin-apriltag/tests/DetectorRegistryTest.cpp`
- Modify: `native/orin-apriltag/CMakeLists.txt`

**Interfaces:**
- Consumes: no CUDA behavior; this task uses a fake `DetectorBackend`.
- Produces: `DetectorRegistry::Create`, `SetCalibration`, `Process`, `Destroy`, and `Size` used by JNI.

- [ ] **Step 1: Define the value types and desired registry API in the failing test**

The header contract must contain:

```cpp
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

using DetectorFactory =
    std::function<std::unique_ptr<DetectorBackend>(int width, int height, int decimate)>;

class DetectorRegistry {
 public:
  explicit DetectorRegistry(DetectorFactory factory);
  std::int64_t Create(int width, int height, int decimate);
  void SetCalibration(std::int64_t handle, const Calibration& calibration);
  std::vector<Detection> Process(std::int64_t handle, const GrayFrame& frame);
  void Destroy(std::int64_t handle);
  std::size_t Size() const;
};
}
```

Write `DetectorRegistryTest.cpp` with a fake backend and assertions for unique handles, invalid handles, injected factory failure leaving `Size()==0`, rejection of a double destroy with `std::invalid_argument` while keeping `Size()==0`, and two threads processing different handles concurrently. Add a `detector_registry_test` CMake target linking only Threads. Create `DetectorRegistry.cpp` containing only `#include "DetectorRegistry.h"` so the first build reaches the linker and proves the methods are absent.

- [ ] **Step 2: Run the native test and observe the missing implementation failure**

On the Jetson:

```bash
cmake -S native/orin-apriltag -B build/orin-apriltag -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build/orin-apriltag --target detector_registry_test
```

Expected: FAIL with missing `DetectorRegistry` symbols.

- [ ] **Step 3: Implement minimal registry synchronization**

Use a registry map protected only for lookup/mutation and a per-slot mutex for backend operations. `Destroy` must erase the handle before waiting for an in-flight call:

```cpp
struct DetectorSlot {
  std::mutex mutex;
  std::unique_ptr<DetectorBackend> detector;
};

std::int64_t DetectorRegistry::Create(int width, int height, int decimate) {
  auto detector = factory_(width, height, decimate);
  if (!detector) throw std::runtime_error("Detector factory returned null");
  auto slot = std::make_shared<DetectorSlot>();
  slot->detector = std::move(detector);
  std::scoped_lock lock(registry_mutex_);
  const std::int64_t handle = next_handle_++;
  slots_.emplace(handle, std::move(slot));
  return handle;
}

void DetectorRegistry::Destroy(std::int64_t handle) {
  std::shared_ptr<DetectorSlot> slot;
  {
    std::scoped_lock lock(registry_mutex_);
    auto it = slots_.find(handle);
    if (it == slots_.end()) throw std::invalid_argument("Invalid CUDA detector handle");
    slot = std::move(it->second);
    slots_.erase(it);
  }
  std::scoped_lock lock(slot->mutex);
  slot->detector.reset();
}
```

`SetCalibration` and `Process` obtain a shared slot under the registry mutex, release the registry mutex, lock the slot, reject a null detector, then call the backend.

- [ ] **Step 4: Run the registry test**

```bash
cmake --build build/orin-apriltag --target detector_registry_test
ctest --test-dir build/orin-apriltag -R detector_registry --output-on-failure
```

Expected: PASS, including the two-handle concurrency assertion.

- [ ] **Step 5: Commit the registry**

```powershell
git add native/orin-apriltag/jni/DetectorRegistry.* native/orin-apriltag/tests/DetectorRegistryTest.cpp native/orin-apriltag/CMakeLists.txt
git commit -m "feat: add safe CUDA detector handle registry"
```

---

### Task 4: Implement the JetPack 7.2 CUDA Backend and Raw JNI

**Files:**
- Create: `native/orin-apriltag/jni/CudaAprilTagDetector.h`
- Create: `native/orin-apriltag/jni/CudaAprilTagDetector.cu`
- Create: `native/orin-apriltag/jni/GpuDetectorJNI.cpp`
- Create: `native/orin-apriltag/tests/CudaDetectorSmokeTest.cpp`
- Create: `native/orin-apriltag/overrides/frc971/orin/cuda.h`
- Modify: `native/orin-apriltag/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 3 `DetectorBackend` and `DetectorRegistry`.
- Produces: `lib971apriltag.so` implementing the Java JNI contract and memory/build diagnostics while leaving the Task 1 vendor manifest valid.

- [ ] **Step 1: Write a failing CUDA smoke test**

Create a test that records `cudaMemGetInfo`, constructs one 1280x800 decimate-2 detector, processes 120 black grayscale frames, destroys it, synchronizes the device, and requires post-destroy free CUDA memory to be within 32 MiB of the baseline. Repeat with two simultaneous detector handles and require both calls to complete without serializing through the registry mutex.

Run:

```bash
cmake --build build/orin-apriltag --target cuda_detector_smoke_test
```

Expected: FAIL because `CudaAprilTagDetector` is missing.

- [ ] **Step 2: Replace non-failing CUDA checks in a staged override**

Copy the pinned `upstream/frc971/orin/cuda.h` to `overrides/frc971/orin/cuda.h`, then replace its logging-only CUDA checks and `assert`-based checks with exceptions. Do not edit any file under `upstream/`; the Task 1 manifest must continue to match the approved upstream revision. The core helper must be:

```cpp
inline void CheckCuda(cudaError_t status, const char* expression,
                      const char* file, int line) {
  if (status == cudaSuccess) return;
  throw std::runtime_error(std::string("CUDA failure: ") + expression +
                           " at " + file + ":" + std::to_string(line) +
                           ": " + cudaGetErrorString(status));
}

#define CHECK_CUDA(expression) \
  ::frc971::apriltag::CheckCuda((expression), #expression, __FILE__, __LINE__)
#define CHECK(condition) \
  do { if (!(condition)) throw std::runtime_error("Check failed: " #condition); } while (false)
#define CHECK_EQ(left, right) CHECK((left) == (right))
#define CHECK_LE(left, right) CHECK((left) <= (right))
#define CHECK_LT(left, right) CHECK((left) < (right))
```

No CUDA allocation may leave an uninitialized pointer after failure.

During CMake configuration, copy `upstream/frc971/orin` into `${CMAKE_CURRENT_BINARY_DIR}/staged/frc971/orin`, overwrite only the staged `cuda.h` with the checked-in override, and compile the CUDA sources from that staged directory. Put the staged include root before the pinned upstream include root so `frc971/orin/...` resolves to the staged CUDA implementation while `third_party/...` still resolves to the pinned source tree.

- [ ] **Step 3: Implement the RAII detector backend**

`CudaAprilTagDetector` owns the tag family, AprilTag detector, GPU detector, and reusable packing buffer. Declare members so reverse destruction order is GPU detector, AprilTag detector, then tag family. Use custom deleters calling `apriltag_detector_destroy` and `tag36h11_destroy`.

Before constructing the FRC 971 GPU detector, set `tag_detector->quad_decimate` to the requested native decimation, set `nthreads=6`, and create the worker pool. The GPU detector constructor checks that its decimation matches the AprilTag detector.

For `Process`:

```cpp
const std::uint8_t* input = frame.data;
if (frame.stride_bytes != static_cast<std::size_t>(frame.width)) {
  packed_.resize(static_cast<std::size_t>(frame.width) * frame.height);
  for (int row = 0; row < frame.height; ++row) {
    std::memcpy(packed_.data() + static_cast<std::size_t>(row) * frame.width,
                frame.data + static_cast<std::size_t>(row) * frame.stride_bytes,
                frame.width);
  }
  input = packed_.data();
}
gpu_detector_->DetectGrayHost(const_cast<std::uint8_t*>(input));
```

Convert each native detection into the Task 3 `Detection` value before returning. The returned values must not borrow native detection memory.

`SetCalibration` destroys the current GPU detector before constructing the calibrated replacement, preventing two large detector allocations from overlapping.

- [ ] **Step 4: Implement raw JNI without WPILib or OpenCV**

Implement JNI methods matching `org.photonvision.jni.GpuDetectorJNI`. Cache a global `AprilTagDetection` class and constructor reference in `JNI_OnLoad`; free the global reference in `JNI_OnUnload`. Delete all local JNI references in the detection loop.

Every value-returning JNI entry point must use this typed boundary helper:

```cpp
template <typename Result, typename Callable>
Result GuardJni(JNIEnv* env, Result failure, Callable&& callable) {
  try {
    return callable();
  } catch (const std::exception& error) {
    jclass exception = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exception, error.what());
    env->DeleteLocalRef(exception);
    return failure;
  } catch (...) {
    jclass exception = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exception, "Unknown CUDA detector failure");
    env->DeleteLocalRef(exception);
    return failure;
  }
}
```

Use `-1` for failed handle creation, `nullptr` for failed object/array/string creation, and `0` for failed memory queries. Add a `GuardJniVoid` overload for calibration and destruction.

Add the shared-library and smoke-test targets to CMake using this exact CUDA source list, where `ORIN_CUDA_SOURCE_DIR` is `${CMAKE_CURRENT_BINARY_DIR}/staged/frc971/orin`:

```cmake
set(ORIN_CUDA_SOURCES
    ${ORIN_CUDA_SOURCE_DIR}/971apriltag.cu
    ${ORIN_CUDA_SOURCE_DIR}/apriltag_detect.cu
    ${ORIN_CUDA_SOURCE_DIR}/labeling_allegretti_2019_BKE.cu
    ${ORIN_CUDA_SOURCE_DIR}/line_fit_filter.cu
    ${ORIN_CUDA_SOURCE_DIR}/points.cu
    ${ORIN_CUDA_SOURCE_DIR}/threshold.cu
    ${ORIN_CUDA_SOURCE_DIR}/cuda.cc
    jni/CudaAprilTagDetector.cu
    jni/DetectorRegistry.cpp
    jni/GpuDetectorJNI.cpp)
```

Link only `CUDA::cudart`, `apriltag_vendor`, JNI, and Threads. Do not link WPILib or OpenCV.

Implement `getCudaFreeMemoryBytes`, `getCudaTotalMemoryBytes`, and `getBuildInfo` using `cudaMemGetInfo` and generated `BuildInfo.h`.

- [ ] **Step 5: Build and run native tests on JetPack 7.2**

Run the vendor-integrity gate on the Windows host before copying sources:

```powershell
powershell -NoProfile -File scripts/tests/verify-orin-cuda-vendor.ps1
```

Then build and inspect the library on the Jetson:

```bash
bash scripts/build-orin-cuda-apriltag.sh
file build/orin-apriltag/lib971apriltag.so
ldd build/orin-apriltag/lib971apriltag.so
```

Expected:

- CTest passes registry and CUDA smoke tests.
- The vendor verifier still passes after the local CUDA-check override is added.
- The library is AArch64.
- `ldd` shows CUDA/runtime system dependencies but no `libwpiutil.so` and no OpenCV library.
- Repeated create/process/destroy returns CUDA free memory within 32 MiB of baseline.

- [ ] **Step 6: Commit the native backend**

```powershell
git add native/orin-apriltag
git commit -m "feat: build CUDA AprilTag JNI for JetPack 7.2"
```

---

### Task 5: Make the Java CUDA Pipe Own Native Lifetime

**Files:**
- Modify: `photon-core/src/main/java/org/photonvision/jni/GpuDetectorJNI.java`
- Modify: `photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagDetectionCudaPipe.java`
- Modify: `photon-core/src/test/java/org/photonvision/vision/pipe/impl/AprilTagDetectionCudaPipeTest.java`

**Interfaces:**
- Consumes: Task 4 JNI signatures.
- Produces: `setEnabled(boolean)`, lazy creation, immediate disable cleanup, single decimation, and raw grayscale calls.

- [ ] **Step 1: Write failing lifecycle and decimation tests**

Add these tests before changing production code:

```java
@Test
void disabledPipeNeverCreatesDetector() {
    var backend = new FakeBackend();
    var pipe = new AprilTagDetectionCudaPipe(backend);
    pipe.setEnabled(false);
    var image = new CVMat(Mat.zeros(480, 640, CvType.CV_8UC1));
    try {
        assertTrue(pipe.run(image).output.isEmpty());
        assertEquals(0, backend.createCount);
    } finally {
        image.release();
        pipe.release();
    }
}

@Test
void disablingDestroysHandleAndReenableCreatesFreshHandle() {
    var backend = new FakeBackend();
    var pipe = new AprilTagDetectionCudaPipe(backend);
    var image = new CVMat(Mat.zeros(480, 640, CvType.CV_8UC1));
    try {
        pipe.setEnabled(true);
        pipe.run(image);
        pipe.setEnabled(false);
        assertEquals(1, backend.destroyCount);
        pipe.setEnabled(true);
        pipe.run(image);
        assertEquals(2, backend.createCount);
    } finally {
        image.release();
        pipe.release();
    }
}

@Test
void nativeDecimateTwoReceivesFullResolutionExactlyOnce() {
    var backend = new FakeBackend();
    var pipe = new AprilTagDetectionCudaPipe(backend);
    pipe.setEnabled(true);
    pipe.setParams(new AprilTagDetectionCudaPipe.AprilTagDetectionCudaPipeParams(2));
    var image = new CVMat(Mat.zeros(1304, 1600, CvType.CV_8UC1));
    try {
        pipe.run(image);
        assertEquals(1600, backend.createdWidth);
        assertEquals(1304, backend.createdHeight);
        assertEquals(2, backend.createdNativeDecimate);
    } finally {
        image.release();
        pipe.release();
    }
}

@Test
void runtimeFailureDestroysHandleBeforeFallbackState() {
    var backend = new FakeBackend();
    backend.failDetection = true;
    var pipe = new AprilTagDetectionCudaPipe(backend);
    pipe.setEnabled(true);
    var image = new CVMat(Mat.zeros(480, 640, CvType.CV_8UC1));
    try {
        assertTrue(pipe.run(image).output.isEmpty());
        assertEquals(1, backend.destroyCount);
        assertFalse(pipe.isAvailable());
    } finally {
        image.release();
        pipe.release();
    }
}
```

Add a matching complete test for decimation `3` that asserts the Java-resized dimensions, native decimation `1`, and one coordinate rescale. Add `destroyFailureCannotBeReenabled` to prove an uncertain native cleanup remains failed until pipeline replacement or service restart. Extend `FakeBackend` to capture data address, width, height, stride, native decimation, CUDA memory values, destroy order, and an injected destroy exception. Update every pre-existing test that expects detector creation to call `pipe.setEnabled(true)` first.

- [ ] **Step 2: Run the focused test and observe expected failures**

```powershell
./gradlew.bat photon-core:test --tests org.photonvision.vision.pipe.impl.AprilTagDetectionCudaPipeTest
```

Expected: FAIL because `setEnabled`, native decimation, raw frame metadata, and immediate failure cleanup are absent.

- [ ] **Step 3: Update the Java JNI facade**

Use these exact native signatures:

```java
public static native long createGpuDetector(int width, int height, int decimate);
public static native void setCalibration(
        long handle, double fx, double fy, double cx, double cy,
        double k1, double k2, double p1, double p2, double k3);
public static native AprilTagDetection[] processGray(
        long handle, long dataAddress, int width, int height, long strideBytes);
public static native void destroyGpuDetector(long handle);
public static native long getCudaFreeMemoryBytes();
public static native long getCudaTotalMemoryBytes();
public static native String getBuildInfo();
```

Retain non-throwing library-load availability reporting.

- [ ] **Step 4: Update the backend and state machine**

The Java backend interface must mirror the JNI methods. Add states `IDLE`, `ACTIVE`, `FAILED`, and `RELEASED`, a `cleanupFailed` flag, plus `setEnabled(boolean)`:

```java
public void setEnabled(boolean enabled) {
    if (state == State.RELEASED) return;
    if (!enabled) {
        this.enabled = false;
        if (cleanupFailed) return;
        try {
            destroyCurrentHandle();
            state = State.IDLE;
        } catch (RuntimeException | LinkageError error) {
            cleanupFailed = true;
            state = State.FAILED;
            logger.error("Failed to disable CUDA AprilTag detector", error);
        }
        return;
    }
    this.enabled = true;
}
```

Creation is allowed only when enabled and idle. A detection exception destroys the current handle before setting `FAILED`. A normal disable resets failure so a later enable can create a fresh handle. `release()` remains terminal and idempotent.

For decimation, use native `1` or `2` with the full frame. For every other value greater than one, perform the existing aligned Java resize and create native decimation `1`. Pass:

```java
backend.detect(
        handle,
        detectionInput.dataAddr(),
        detectionInput.cols(),
        detectionInput.rows(),
        detectionInput.step1());
```

The frame is guaranteed grayscale `CV_8UC1`, so `step1()` is the row stride in bytes.

- [ ] **Step 5: Add bounded diagnostics**

Log native build info once after successful load. Log free/total CUDA memory before creation, after creation, and after destruction. Do not log every frame. A normal disable resets failure only after destruction succeeds; `cleanupFailed` is terminal for that pipe and prevents subsequent settings updates from pretending cleanup succeeded.

- [ ] **Step 6: Run Java tests green**

```powershell
./gradlew.bat photon-core:test --tests org.photonvision.vision.pipe.impl.AprilTagDetectionCudaPipeTest
```

Expected: all pipe tests pass with no leaked `CVMat` warning.

- [ ] **Step 7: Commit the Java lifecycle repair**

```powershell
git add photon-core/src/main/java/org/photonvision/jni/GpuDetectorJNI.java photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagDetectionCudaPipe.java photon-core/src/test/java/org/photonvision/vision/pipe/impl/AprilTagDetectionCudaPipeTest.java
git commit -m "fix: bound CUDA detector native lifetime"
```

---

### Task 6: Wire Enable/Disable into the AprilTag Pipeline

**Files:**
- Modify: `photon-core/src/main/java/org/photonvision/vision/pipeline/AprilTagPipeline.java`
- Modify: `photon-core/src/main/java/org/photonvision/vision/pipeline/AprilTagPipelineSettings.java`
- Modify: `photon-core/src/test/java/org/photonvision/vision/pipeline/AprilTagPipelineMLBehaviorTest.java`
- Modify: `photon-client/src/components/dashboard/tabs/AprilTagTab.vue`
- Modify: `photon-client/src/types/PipelineTypes.ts`

**Interfaces:**
- Consumes: Task 5 `AprilTagDetectionCudaPipe.setEnabled(boolean)`.
- Produces: per-frame settings propagation and tested CUDA-to-TensorRT-to-CPU fallback.

- [ ] **Step 1: Add failing pipeline behavior tests**

Add tests proving:

- CUDA is enabled only for `useCudaTagDetection=true` and tag36h11.
- Changing the setting to false calls `setEnabled(false)` before the next frame is processed.
- A failed CUDA pipe invokes TensorRT ROI when enabled.
- A failed CUDA pipe invokes CPU when TensorRT ROI is disabled.
- A valid empty CUDA result invokes neither fallback.
- New AprilTag settings default `useCudaTagDetection` to false.

The fake CUDA pipe must record enable transitions and expose a failed state without allocating JNI resources.

- [ ] **Step 2: Run tests red**

```powershell
./gradlew.bat photon-core:test --tests org.photonvision.vision.pipeline.AprilTagPipelineMLBehaviorTest
```

Expected: the disable-transition assertion fails because the current pipeline never calls `setEnabled(false)`.

- [ ] **Step 3: Propagate desired state from settings**

At the start of `setPipeParamsImpl`, before CUDA params are applied, add:

```java
boolean cudaEnabled =
        settings.useCudaTagDetection && settings.tagFamily == AprilTagFamily.kTag36h11;
cudaDetectionPipe.setEnabled(cudaEnabled);
```

Only call `setParams` when `cudaEnabled` is true. Keep the existing process selection order and valid-empty semantics.

- [ ] **Step 4: Run the combined detector tests**

```powershell
./gradlew.bat photon-core:test \
  --tests org.photonvision.vision.pipe.impl.AprilTagDetectionCudaPipeTest \
  --tests org.photonvision.vision.pipeline.AprilTagPipelineMLBehaviorTest
```

Expected: both suites pass.

Run the client checks to preserve the existing CUDA toggle and default:

```powershell
corepack pnpm --dir photon-client typecheck
node --check photon-client/scripts/set-photonvision-pipeline.mjs
```

- [ ] **Step 5: Commit pipeline integration**

```powershell
git add photon-core/src/main/java/org/photonvision/vision/pipeline/AprilTagPipeline.java photon-core/src/test/java/org/photonvision/vision/pipeline/AprilTagPipelineMLBehaviorTest.java photon-client/src/components/dashboard/tabs/AprilTagTab.vue photon-client/src/types/PipelineTypes.ts
git add -p photon-core/src/main/java/org/photonvision/vision/pipeline/AprilTagPipelineSettings.java
git commit -m "fix: release CUDA detector when pipeline disables it"
```

Stage only CUDA-setting hunks from `AprilTagPipelineSettings.java`; preserve the unrelated auto-exposure change for its own commit.

---

### Task 7: Add Orin Service Limits and Atomic Native Deployment

**Files:**
- Modify: `scripts/orin-nvme-team-image.sh`
- Create: `scripts/tests/orin-nvme-cuda-service-test.sh`

**Interfaces:**
- Consumes: Task 4 `lib971apriltag.so` and existing `--cuda-apriltag-library` option.
- Produces: `MemoryHigh=5G`, `MemoryMax=6G`, atomic native installation, restart, and HTTP health check.

- [ ] **Step 1: Write a failing shell contract test**

Create `scripts/tests/orin-nvme-cuda-service-test.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
script="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/orin-nvme-team-image.sh"
bash -n "${script}"
grep -q '^MemoryHigh=5G$' "${script}"
grep -q '^MemoryMax=6G$' "${script}"
grep -q 'systemctl stop photonvision.service' "${script}"
grep -q 'curl.*127.0.0.1:5800' "${script}"
```

- [ ] **Step 2: Run the test red**

```bash
bash scripts/tests/orin-nvme-cuda-service-test.sh
```

Expected: FAIL because the memory limits and live health check are absent.

- [ ] **Step 3: Add the service memory limits**

Add under `[Service]` in the generated unit:

```ini
MemoryHigh=5G
MemoryMax=6G
```

Keep `Restart=always` and `RestartSec=2`.

- [ ] **Step 4: Make native replacement atomic**

When `--cuda-apriltag-library` is present and PhotonVision is active:

1. Stop `photonvision.service`.
2. Install to `${PV_DIR}/lib/lib971apriltag.so.new` with mode `0755`.
3. Rename the staged file to `lib971apriltag.so` on the same filesystem.
4. Run `systemctl daemon-reload` and start PhotonVision.
5. Poll `http://127.0.0.1:5800/` for up to 45 seconds.
6. Fail provisioning if HTTP never returns 200.

Do not modify the UVC or deferred-device defaults.

- [ ] **Step 5: Run syntax and contract tests**

```bash
bash -n scripts/orin-nvme-team-image.sh
bash scripts/tests/orin-nvme-cuda-service-test.sh
```

Expected: PASS.

- [ ] **Step 6: Commit service protection**

```powershell
git add scripts/orin-nvme-team-image.sh scripts/tests/orin-nvme-cuda-service-test.sh
git commit -m "build: protect Orin from native memory exhaustion"
```

---

### Task 8: Add the Live Memory Acceptance Monitor

**Files:**
- Create: `scripts/monitor-orin-photonvision.sh`
- Create: `scripts/tests/monitor-orin-photonvision-test.sh`
- Modify: `photon-client/scripts/benchmark-photonvision.mjs`

**Interfaces:**
- Consumes: systemd service state and `/proc` metrics on the Jetson.
- Produces: timestamped CSV plus a nonzero exit when memory, service, or availability gates fail.

- [ ] **Step 1: Write failing threshold tests**

The monitor must support `--samples-file <fixture>` so threshold logic can be tested without waiting. Create fixtures in the test script for:

- RSS growth of 64 MiB: pass.
- RSS growth of 256 MiB: fail.
- available memory of 1.4 GiB: fail.
- restart-count increase: fail.
- stable samples with 2 GiB available: pass.

Run:

```bash
bash scripts/tests/monitor-orin-photonvision-test.sh
```

Expected: FAIL because the monitor is absent.

- [ ] **Step 2: Implement monitor output and thresholds**

`scripts/monitor-orin-photonvision.sh` must accept:

```text
--duration-seconds <n>
--interval-seconds <n>
--output <csv>
--max-rss-growth-mib <n>
--min-available-mib <n>
--samples-file <csv>
```

Each live row contains monotonic seconds, Java PID, RSS KiB, available KiB, service restart count, active state, and HTTP status. The script exits immediately if PhotonVision is inactive, HTTP fails twice consecutively, available memory crosses the minimum, or restart count changes. RSS growth is evaluated after a 60-second warmup.

- [ ] **Step 3: Expose all detector flags in websocket benchmarks**

Add `useMLDetection` and `cameraAutoExposure` to the camera summary in `benchmark-photonvision.mjs`. Keep its existing FPS, latency, tag-hit, and MultiTag-hit calculations. Capture native build information from PhotonVision's one-time startup log instead of adding a new websocket contract.

- [ ] **Step 4: Run tests and static checks**

```bash
bash scripts/tests/monitor-orin-photonvision-test.sh
bash -n scripts/monitor-orin-photonvision.sh
node --check photon-client/scripts/benchmark-photonvision.mjs
```

Expected: PASS.

- [ ] **Step 5: Commit monitoring**

```powershell
git add scripts/monitor-orin-photonvision.sh scripts/tests/monitor-orin-photonvision-test.sh photon-client/scripts/benchmark-photonvision.mjs
git commit -m "test: monitor Orin CUDA memory stability"
```

---

### Task 9: Build, Deploy, and Validate One CUDA Camera

**Files:**
- Modify after successful measurement: `docs/orin-nano-nvme-team-image.md`
- Runtime artifact: `/opt/photonvision/lib/lib971apriltag.so`
- Runtime artifact: `/opt/photonvision/photonvision.jar`

**Interfaces:**
- Consumes: Tasks 1-8 and fresh calibrations for both active Thrifty cameras.
- Produces: a five-minute one-camera stability result with and without pose/MultiTag.

- [ ] **Step 1: Back up settings and reject mismatched archived calibration**

Download the current device configuration:

```powershell
curl.exe -o build/orin-pre-cuda-config.zip http://192.168.50.97:5800/api/settings/photonvision_config.zip
```

Do not import `../photonvision-ml-tag-experimental-settings-export.zip` into the live Orin database. The archive contains calibrations for camera UUIDs `21ba7c05-8513-4b27-9ff0-47fb3278032a` and `5add64a1-abdd-4b62-a19a-07bc841af219`, at 320x240, 640x480, 1280x720, and 1600x1304. The active Thrifty camera UUIDs are different and their selected mode is 1280x800, so transplanting those intrinsics would invalidate pose measurements and the full settings import could overwrite Orin configuration.

- [ ] **Step 2: Build and test the native library on the Jetson**

Copy the repository native tree and build script to `/home/doug/photonvision-cuda-src`, then run:

```powershell
$revision = git rev-parse --short=12 HEAD
ssh doug@192.168.50.97 "mkdir -p /home/doug/photonvision-cuda-src/native /home/doug/photonvision-cuda-src/scripts"
scp -r native/orin-apriltag doug@192.168.50.97:/home/doug/photonvision-cuda-src/native/
scp scripts/build-orin-cuda-apriltag.sh doug@192.168.50.97:/home/doug/photonvision-cuda-src/scripts/
ssh doug@192.168.50.97 "cd /home/doug/photonvision-cuda-src && bash scripts/build-orin-cuda-apriltag.sh --source-revision $revision"
```

Then run on the Jetson:

```bash
bash scripts/build-orin-cuda-apriltag.sh
ctest --test-dir build/orin-apriltag --output-on-failure
```

Expected: all native tests pass and `ldd` has no missing libraries.

- [ ] **Step 3: Build the Linux ARM64 JAR on Windows**

```powershell
./gradlew.bat photon-targeting:jar photon-server:shadowJar -PArchOverride=linuxarm64
```

Expected: `BUILD SUCCESSFUL` and one `photonvision-*-linuxarm64.jar` in `photon-server/build/libs`.

- [ ] **Step 4: Install both artifacts with CUDA disabled**

Copy the JAR, stop PhotonVision, install the Jetson-built native library and copied JAR atomically, and start the service:

```powershell
$jar = Get-ChildItem photon-server/build/libs/*linuxarm64.jar | Sort-Object LastWriteTime -Descending | Select-Object -First 1
scp $jar.FullName doug@192.168.50.97:/tmp/photonvision-linuxarm64.jar
```

```bash
sudo systemctl stop photonvision
sudo install -m 0755 /home/doug/photonvision-cuda-src/build/orin-apriltag/lib971apriltag.so /opt/photonvision/lib/lib971apriltag.so.new
sudo mv /opt/photonvision/lib/lib971apriltag.so.new /opt/photonvision/lib/lib971apriltag.so
sudo install -m 0644 /tmp/photonvision-linuxarm64.jar /opt/photonvision/photonvision.jar.new
sudo mv /opt/photonvision/photonvision.jar.new /opt/photonvision/photonvision.jar
sudo systemctl start photonvision
```

Verify:

```bash
systemctl is-active photonvision
curl --fail http://127.0.0.1:5800/
systemctl show photonvision -p MemoryHigh -p MemoryMax -p NRestarts
```

Expected: active, HTTP 200, `MemoryHigh=5368709120`, `MemoryMax=6442450944`, and zero restarts since deployment.

- [ ] **Step 5: Run one camera without pose**

Use the active camera IDs discovered by `benchmark-photonvision.mjs`. On the current device they are:

- `b65c14de-6839-4d99-a2d3-0f2952be8794`
- `41785947-bb90-4f32-8448-6d7afe9b3954`

Set the first to `useCudaTagDetection=true`, `solvePNPEnabled=false`, `doMultiTarget=false`; keep the second CUDA setting false. Start the five-minute monitor with 128 MiB maximum RSS growth and 1536 MiB minimum available memory, then run the websocket benchmark for the same interval.

Use separate terminals for the concurrent commands:

```bash
bash scripts/monitor-orin-photonvision.sh \
  --duration-seconds 300 --interval-seconds 5 \
  --output /tmp/orin-single-cuda-memory.csv \
  --max-rss-growth-mib 128 --min-available-mib 1536
```

```powershell
corepack pnpm --dir photon-client exec node scripts/benchmark-photonvision.mjs 192.168.50.97:5800 300 60
```

Expected: no restarts, no OOM, no `CVMat` warnings, tag hit rate at least 99%, and memory plateaus after warmup.

- [ ] **Step 6: Run one camera with pose and MultiTag**

Before this step, calibrate the first physical Thrifty camera at exactly 1280x800 in PhotonVision and export its calibration JSON for backup. Confirm this endpoint returns HTTP 200 for the active UUID:

```powershell
curl.exe --fail -o build/thrifty-camera-1-1280x800.json "http://192.168.50.97:5800/api/utils/getCalibrationJSON?cameraUniqueName=b65c14de-6839-4d99-a2d3-0f2952be8794&width=1280&height=800"
```

Then set `solvePNPEnabled=true` and `doMultiTarget=true` on the first camera. Repeat the five-minute monitor and benchmark.

Expected: stability gates remain satisfied and MultiTag hit rate is at least 95% while tags 3 and 4 remain visible.

- [ ] **Step 7: Exercise disable and re-enable cleanup**

Disable CUDA on the first camera and confirm CUDA free memory rises and process RSS does not require a service restart. Re-enable it and run a one-minute benchmark.

Expected: a fresh detector is created, tags remain detected, and no handle-limit or stale-handle error appears.

- [ ] **Step 8: Record the one-camera evidence**

Add a dated table to `docs/orin-nano-nvme-team-image.md` containing input mode, detector mode, pose flags, average/p95 FPS, average/p95 latency, tag hit, MultiTag hit, start/end RSS, minimum available memory, and restart count.

Do not claim dual-camera validation yet.

---

### Task 10: Validate Two CUDA Cameras and Publish Results

**Files:**
- Modify: `docs/orin-nano-nvme-team-image.md`

**Interfaces:**
- Consumes: Task 9 stable one-camera artifacts and calibrated active cameras.
- Produces: final ten-minute dual-camera acceptance evidence and corrected deployment documentation.

- [ ] **Step 1: Enable CUDA, pose, and MultiTag on both active cameras**

Calibrate the second physical Thrifty camera independently at 1280x800 and export its JSON:

```powershell
curl.exe --fail -o build/thrifty-camera-2-1280x800.json "http://192.168.50.97:5800/api/utils/getCalibrationJSON?cameraUniqueName=41785947-bb90-4f32-8448-6d7afe9b3954&width=1280&height=800"
```

Use `set-photonvision-pipeline.mjs` and verify a fresh websocket connection reports all three flags true for both active cameras:

```powershell
$settings = '{"useCudaTagDetection":true,"solvePNPEnabled":true,"doMultiTarget":true}'
Push-Location photon-client
node scripts/set-photonvision-pipeline.mjs 192.168.50.97:5800 b65c14de-6839-4d99-a2d3-0f2952be8794 $settings
node scripts/set-photonvision-pipeline.mjs 192.168.50.97:5800 41785947-bb90-4f32-8448-6d7afe9b3954 $settings
Pop-Location
```

Confirm the stale `PC_Camera` entries still produce no samples and exclude them from results.

- [ ] **Step 2: Start the mandatory ten-minute monitor**

Run on the Jetson:

```bash
bash scripts/monitor-orin-photonvision.sh \
  --duration-seconds 600 \
  --interval-seconds 5 \
  --output /tmp/orin-dual-cuda-memory.csv \
  --max-rss-growth-mib 128 \
  --min-available-mib 1536
```

At the same time, run from Windows:

```powershell
corepack pnpm --dir photon-client exec node scripts/benchmark-photonvision.mjs 192.168.50.97:5800 600 60
```

- [ ] **Step 3: Check acceptance evidence**

Require all of the following before proceeding:

- No service restart, OOM record, JNI exception, camera disconnect, or unreleased `CVMat` warning.
- RSS growth no greater than 128 MiB after warmup.
- At least 1536 MiB available system memory throughout.
- At least 99% tag hit rate on both active cameras.
- At least 95% MultiTag hit rate while tags 3 and 4 remain visible.
- At least 36 average processed FPS per 1280x800 camera.

If stability passes but performance misses 36 FPS, stop here and open a separate profiling cycle. Do not weaken memory or detection gates.

- [ ] **Step 4: Verify the 50 FPS stretch target separately**

Only after Step 3 passes, record GPU utilization and test stream-output suppression, decimation, and camera input mode one variable at a time. The stretch result requires 50 average FPS per camera and p95 latency below 50 ms without reducing tag or MultiTag hit rates.

- [ ] **Step 5: Replace stale documentation claims with measured values**

Update `docs/orin-nano-nvme-team-image.md` to include:

- Native build command and embedded build-info output.
- Required JetPack development packages.
- Service memory limits.
- One-camera and two-camera measurement tables.
- JAR/library-only deployment workflow.
- Explicit distinction between the 36 FPS minimum and 50 FPS stretch target.

- [ ] **Step 6: Run final repository verification**

```powershell
./gradlew.bat photon-core:test \
  --tests org.photonvision.vision.pipe.impl.AprilTagDetectionCudaPipeTest \
  --tests org.photonvision.vision.pipeline.AprilTagPipelineMLBehaviorTest \
  --tests org.photonvision.vision.pipeline.AprilTagPipelineSettingsTest
./gradlew.bat photon-targeting:jar photon-server:shadowJar -PArchOverride=linuxarm64
node --check photon-client/scripts/benchmark-photonvision.mjs
git diff --check
```

On the Jetson:

```bash
ctest --test-dir build/orin-apriltag --output-on-failure
bash scripts/tests/monitor-orin-photonvision-test.sh
bash scripts/tests/orin-nvme-cuda-service-test.sh
```

Expected: every command passes.

- [ ] **Step 7: Commit validated documentation and any final test-only adjustments**

```powershell
git add docs/orin-nano-nvme-team-image.md
git commit -m "docs: record JetPack 7.2 CUDA validation"
```

Do not include unrelated camera, networking, model, or formatting changes in this commit.
