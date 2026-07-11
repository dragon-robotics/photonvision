# JetPack 7.2 CUDA AprilTag Detector Design

**Status:** Approved

**Date:** 2026-07-11

**Branch:** `image/orin-nano`

**Target:** Jetson Orin Nano 8 GB, JetPack 7.2 / L4T R39.2, Java 17,
PhotonVision 2026.3.4

## Problem

The current Orin CUDA integration loads a prebuilt `lib971apriltag.so` derived
from FRC Team 4143's JetPack 6.2-oriented `GpuDetectorJNI` project. Enabling
CUDA detection on both active 1280x800 USB cameras caused PhotonVision to stop
responding. The kernel then killed Java for exhausting system memory at about
7 GB resident memory. Disabling CUDA did not release the native detectors;
memory returned to normal only after restarting PhotonVision.

The current integration therefore has two separate defects:

1. The Java pipeline does not destroy an active native detector when CUDA is
   disabled.
2. The native binary is not reproducibly built, instrumented, or validated for
   JetPack 7.2, so its allocation and failure behavior cannot be trusted.

The existing Orin image documentation claims successful JetPack 7.2 CUDA
validation. That claim must be replaced with results from the acceptance tests
in this design.

## Goals

- Build the CUDA AprilTag detector reproducibly on JetPack 7.2.
- Support two concurrent 1280x800 USB camera pipelines.
- Preserve the existing PhotonVision pose-estimation and MultiTag code paths.
- Release all native resources when CUDA is disabled, settings change, the
  camera mode changes, or the pipeline is released.
- Fall back to TensorRT ROI detection or CPU AprilTag detection after a native
  failure without crashing PhotonVision.
- Keep native and process memory bounded during sustained operation.
- Reach at least 36 processed FPS per USB camera, with 50 FPS per camera as the
  stretch target.

## Non-Goals

- Adding Jetson CSI or Isaac ROS camera support.
- Reworking the TensorRT ROI model or its inference path.
- Changing USB bandwidth quirks, camera matching, or exposure behavior.
- Supporting CUDA AprilTag detection on non-Orin platforms in this phase.
- Producing a cross-compiled CUDA binary on Windows.

## Chosen Approach

Vendor and harden the 4143 CUDA detector source inside this PhotonVision
branch. Build it directly on the Jetson with the JetPack 7.2 CUDA toolkit and
JDK 17 headers. The source will live under `native/orin-apriltag` with its
upstream Apache and third-party license files and a document recording the
pinned upstream revision.

This approach was selected over two alternatives:

- A separate fork and release artifact would preserve repository separation,
  but it would require coordinating two repositories and a Jetson-capable
  release pipeline for every native change.
- A separate CUDA helper process would provide stronger crash isolation, but
  shared-memory transport, process supervision, and serialization would add
  substantial complexity before the native lifetime problem is understood.

## Source Layout

The implementation will add these focused units:

- `native/orin-apriltag/`: pinned and licensed CUDA/AprilTag source.
- `native/orin-apriltag/jni/`: raw JNI adapter and detector registry.
- `native/orin-apriltag/tests/`: native registry, allocation, and smoke tests.
- `scripts/build-orin-cuda-apriltag.sh`: target-side JetPack 7.2 build entry
  point.
- Existing `scripts/orin-nvme-team-image.sh`: installation and systemd memory
  protection.
- Existing Java CUDA pipe and tests: lifecycle, fallback, and detector
  selection behavior.

The native source will not depend on a separately built allwpilib tree or a
system-installed `libwpiutil.so`.

## JNI Contract

The JNI facade will expose a small synchronous API:

```java
long createGpuDetector(int width, int height, int decimate);
void setCalibration(
        long handle,
        double fx,
        double fy,
        double cx,
        double cy,
        double k1,
        double k2,
        double p1,
        double p2,
        double k3);
AprilTagDetection[] processGray(
        long handle,
        long dataAddress,
        int width,
        int height,
        long strideBytes);
void destroyGpuDetector(long handle);
long getCudaFreeMemoryBytes();
long getCudaTotalMemoryBytes();
String getBuildInfo();
```

The JNI layer will use raw JNI APIs instead of `wpi/jni_util.h`. It will receive
the grayscale Mat data address, dimensions, and row stride rather than a
`cv::Mat*`, eliminating the C++ OpenCV ABI dependency. Calls are synchronous,
so the Mat remains alive for the complete native operation. Non-contiguous
input is packed into a detector-owned reusable host buffer; no buffer is
allocated per frame.

`getBuildInfo()` will report the source revision, L4T release, CUDA compiler
version, CUDA architecture, and build timestamp.

## Native Lifetime and Concurrency

Native handles will be generated IDs stored in a validated registry. The
registry contains `shared_ptr<DetectorState>` values and is protected by a
short-held registry mutex. Each detector state has its own mutex, CUDA stream,
AprilTag detector, tag family, reusable host buffer, and GPU detector.

Detection on different camera handles runs concurrently. The registry mutex is
used only to obtain a state reference; it is never held while CUDA work runs.
The per-detector mutex serializes detection, calibration replacement, and
destruction for that camera.

Destroying a handle removes it from the registry first, then waits for any
in-flight operation and releases the detector state. Invalid, stale, duplicate,
and already-destroyed handles produce a Java exception rather than indexing a
fixed global array.

All resources use RAII. Destruction includes:

- CUDA buffers, events, and streams.
- GPU detector state.
- AprilTag detections from the previous frame.
- AprilTag worker pools and detector structures.
- The tag36h11 family.
- Reusable host packing buffers.

Partial construction cleans up every resource acquired before the failure and
does not publish a registry handle.

## Decimation and Frame Changes

Native decimation values `1` and `2` are passed directly to the corresponding
GPU detector constructor. The Java pipe sends the full-resolution frame, so
those values are never applied twice.

For other PhotonVision decimation values, Java performs one resize and creates
the native detector with decimation `1`. Returned corners and homographies are
scaled back to the original frame coordinates once.

A width, height, decimation, or calibration change destroys the old detector
before constructing the replacement. This avoids transiently holding two
large detector allocations for one camera. A failed replacement leaves the
pipe without a native handle and activates the normal fallback path.

## Java Pipeline Lifecycle

`AprilTagDetectionCudaPipe` will have explicit idle, active, failed, and
released states.

- Enabling CUDA permits lazy detector creation on the first valid grayscale
  frame.
- Disabling CUDA immediately destroys an active native handle and returns the
  pipe to idle.
- Re-enabling CUDA after a normal disable creates a fresh detector.
- A native runtime failure destroys the handle, marks the pipe failed for that
  pipeline instance, logs CUDA memory diagnostics, and returns control to the
  detector-selection fallback.
- Releasing the pipeline is terminal and idempotent.

Detector selection remains:

1. CUDA full-frame detector when enabled and healthy for tag36h11.
2. TensorRT ML ROI detector when enabled and available.
3. Traditional CPU AprilTag detector.

A valid empty CUDA result means no tags were found and does not invoke a slower
fallback. A CUDA exception or unavailable backend does invoke the fallback.

## Native Error Handling

The native library will not use `assert` or continue after failed CUDA calls.
CUDA errors, invalid dimensions, unsupported decimation, invalid stride,
invalid data pointers, and invalid handles become descriptive Java exceptions.
JNI catches every C++ exception before it crosses the native boundary.

CUDA free and total memory are logged before detector construction, after
successful construction, and after destruction. Allocation failures include
the requested detector dimensions and decimation in the error message.

## Process Memory Protection

The Orin team image will add these service limits for the 8 GB target:

```ini
MemoryHigh=5G
MemoryMax=6G
```

The existing `Restart=always` policy restarts PhotonVision if a future native
regression exceeds the hard limit. These limits are defense in depth, not a
substitute for bounded native allocation.

## JetPack 7.2 Build

`scripts/build-orin-cuda-apriltag.sh` will run on the Jetson and verify:

- AArch64 architecture.
- L4T R39.2 / JetPack 7.2.
- JDK 17 headers.
- CMake and a working CUDA compiler.
- CUDA compute architecture 8.7 for Orin.

The build uses the vendored AprilTag and FRC 971 CUDA sources and produces a
versioned `lib971apriltag.so`. It does not download or compile allwpilib. The
provisioning script installs the library into `/opt/photonvision/lib` while
PhotonVision is stopped, starts the service, and verifies the HTTP endpoint.

Later agentic iterations can rebuild and replace only the native library and
JAR without reflashing NVMe.

## Testing Strategy

### Java Tests

Fake-backend tests will verify:

- Lazy creation at real frame dimensions.
- Enable, disable, re-enable, and idempotent release.
- Destruction before resolution, calibration, or decimation replacement.
- Correct native-versus-Java decimation ownership and output scaling.
- Valid empty CUDA results without fallback.
- Native creation and runtime failures with TensorRT or CPU fallback.
- Independent lifecycle for two camera pipes.

Every production behavior change follows red-green-refactor: its test must be
observed failing before the implementation is added.

### Native Tests

Jetson-native tests will verify:

- Invalid and stale handle rejection.
- Partial-construction cleanup after injected allocation failures.
- Correct destruction of detector, worker pool, and tag family resources.
- Concurrent calls on two independent detector handles.
- Safe serialization of process versus calibration or destruction on one
  handle.
- Non-contiguous grayscale input through the reusable packing buffer.
- Stable CUDA free-memory readings across repeated create/process/destroy
  cycles.

### Live Validation

Validation uses the existing PhotonVision websocket benchmark plus process RSS,
system memory, CUDA memory, service restart count, and kernel logs.

1. Record a clean CPU baseline after restarting PhotonVision.
2. Run one 1280x800 camera with CUDA detection and pose disabled for five
   minutes.
3. Repeat one camera with calibration, pose estimation, and MultiTag enabled.
4. Run both 1280x800 cameras with CUDA, pose estimation, and MultiTag for at
   least ten minutes.

The test immediately disables CUDA and restarts PhotonVision if RSS or CUDA
memory grows without a plateau, available system memory falls below 1.5 GiB,
or either service becomes unresponsive.

## Acceptance Criteria

- No OOM kills, service restarts, JNI failures, unreleased `CVMat` warnings, or
  camera disconnects during the final dual-camera run.
- After a one-minute warmup, PhotonVision RSS grows by no more than 128 MB over
  ten minutes.
- At least 1.5 GiB system memory remains available.
- Continuously visible tags are detected in at least 99 percent of processed
  samples.
- MultiTag succeeds in at least 95 percent of samples while multiple mapped
  tags remain visible.
- Both 1280x800 USB cameras sustain at least 36 processed FPS.
- The stretch target is 50 processed FPS per camera with p95 latency below
  50 ms.

Failure to reach 36 FPS after stability passes starts a separate profiling and
optimization cycle; it does not justify weakening the memory or detection
criteria.

## Documentation and Rollout

The Orin image documentation will explain the native build command, installed
library metadata, service memory limits, health checks, and JAR/library-only
iteration workflow. Previous JetPack 7.2 benchmark claims will be removed until
the final acceptance run produces replacement evidence.

The rollout order is one camera without pose, one camera with pose and
MultiTag, then two cameras. CUDA remains disabled by default until all final
acceptance criteria pass.
