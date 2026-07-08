# Orin Nano ML AprilTag Image Design

## Context

This branch, `image/orin-nano`, is based on PhotonVision `v2026.3.4`. The goal is to create a bootable PhotonVision image for NVIDIA Jetson Orin Nano hardware and add ML-assisted AprilTag ROI detection using the Orin Nano CUDA GPU.

The implementation will use selective backports only. The `FRC-Team-4143/photonvision` `jetson-orin` branch is a reference for Jetson platform and CUDA AprilTag work, but it contains unrelated changes and will not be merged wholesale. The `DoctorFogarty/photonvision` `apriltag-ml-experimental` branch is a reference for ML ROI detection and ROI decode flow, but it will not be merged wholesale. The local `2026-team2375` branch is the reference for static networking and manual exposure startup fixes.

## Goals

- Build a bootable Orin Nano PhotonVision image artifact from this branch.
- Keep the Java 17 and PhotonVision `v2026.3.4` base intact.
- Make Team 2375 networking the global default for all artifacts built from `image/orin-nano`.
- Use static IP `10.23.75.15` with gateway `10.23.75.4`.
- Set default hostname to `photonvision-orin-nano-ml-tag`.
- Fix first-boot manual exposure application so cameras do not require toggling auto exposure in the UI.
- Add ML-assisted AprilTag ROI detection using the Orin Nano CUDA GPU.
- Preserve traditional AprilTag detection as a fallback path.
- Keep Rubik Pi 3 and Orange Pi NPU-specific behavior out of this branch except for shared abstractions needed by the ROI pipeline.

## Non-Goals

- Do not add Wi-Fi fallback to this image.
- Do not merge the full 4143 Jetson branch.
- Do not merge the full DoctorFogarty ML branch.
- Do not change the Rubik Pi 3 image or Team 2375 Rubik workflow.
- Do not require users to manually toggle camera exposure after boot.
- Do not depend on Rubik TFLite delegate or RKNN for Orin inference.

## Approach

Use a selective backport with one new Orin platform path.

The Orin image will be built as a `linuxarm64` PhotonVision JAR plus an Orin Nano base image that provides the NVIDIA userspace stack. The image build step will install the JAR, install a NetworkManager static Ethernet profile, set up PhotonVision startup, clear stale persisted PhotonVision network settings, and package the modified image artifact.

The ML feature will reuse the normal AprilTag pipeline instead of adding a separate user-facing pipeline type. A new CUDA/TensorRT-backed object detector will feed AprilTag ROI boxes to the existing ROI decode path. The regular WPILib AprilTag detector will then decode the tag IDs and corners inside those ROIs, preserving pose-estimation compatibility with the current AprilTag pipeline.

## Image Build

Add an Orin Nano image matrix entry to `.github/workflows/build.yml`.

The matrix entry will:

- Use the `LinuxArm64` JAR artifact.
- Use an Orin Nano base image URL pinned in the workflow or branch configuration.
- Use a new image suffix such as `orinnano`.
- Use platform override `LINUX_AARCH64` or a new Jetson-specific platform enum if needed.
- Run `scripts/armrunner.sh` or a small Orin-specific runner script to install PhotonVision.
- Upload `photonvision*_orinnano.img.xz`.

The runner script will install:

- `/opt/photonvision/photonvision.jar`
- A PhotonVision service if the base image does not already include one.
- A NetworkManager Ethernet profile for static IP `10.23.75.15/8` and gateway `10.23.75.4`.
- Hostname `photonvision-orin-nano-ml-tag`.

It will not install a WLAN profile.

The image should clear stale persisted network configuration from the base image before first boot:

- `/opt/photonvision/photonvision_config/photon.sqlite*`
- `/opt/photonvision/photonvision_config/networkSettings.json`

This keeps the global branch defaults and baked NetworkManager profile from being hidden by an old base-image config database.

## Network Defaults

Change `NetworkConfig` defaults on this branch:

- `ntServerAddress = "2375"`
- `connectionType = NetworkMode.STATIC`
- `staticIp = "10.23.75.15"`
- `hostname = "photonvision-orin-nano-ml-tag"`

The NetworkManager profile baked into the Orin image will use:

- Address: `10.23.75.15/8`
- Gateway: `10.23.75.4`
- IPv6: disabled
- Autoconnect: enabled
- Autoconnect priority: high

The early interface selection fix from `2026-team2375` will be adapted. PhotonVision should configure the first wired interface even before NetworkManager reports an active connection, avoiding the boot race where a static IP is not reachable until the UI or service state changes.

## Exposure Startup Fix

Backport the camera settings ordering fix from `2026-team2375`.

Current risk: on first boot, manual exposure can be ignored until the user toggles auto exposure in the UI. The fix is to prime the manual exposure value before switching the camera into manual exposure mode.

The pipeline application order will be:

1. Set video mode.
2. Set brightness.
3. If `cameraAutoExposure` is false, validate and set `cameraExposureRaw`.
4. Set auto exposure to the configured state.
5. Apply gain and white balance settings.

This keeps the saved exposure available for USB camera quirk logic that briefly toggles auto exposure to force the camera to accept manual exposure.

## Platform Detection

Add Jetson/Orin detection to `Platform`.

Detection should key off stable Linux device-tree strings, such as `/proc/device-tree/model` containing `NVIDIA Jetson`, then expose a helper such as `Platform.isJetson()`.

If a dedicated enum is added, prefer a narrow name such as `LINUX_JETSON_ORIN_NANO` only if the code needs to distinguish Orin Nano from generic AArch64. If the image and backend only need Linux AArch64 plus Jetson detection, keep `LINUX_AARCH64` and add helper methods to reduce platform enum churn.

The UI general settings should show CUDA/TensorRT acceleration when the backend is available.

## ML Backend

Add a CUDA/TensorRT object detection backend behind the existing `Model` and `ObjectDetector` interfaces.

The backend will:

- Run only on Jetson/Orin platforms.
- Use the CUDA GPU through TensorRT or a TensorRT-backed JNI library.
- Accept an AprilTag YOLO model source suitable for Orin.
- Produce `NeuralNetworkPipeResult` boxes in full-frame coordinates after letterbox scaling is reversed.
- Release native resources reliably.
- Return an empty result instead of crashing when the native backend or model is unavailable.

Use a model family such as `CUDA` or `TENSORRT` in `NeuralNetworkModelManager`. The selected family must be exposed through `supportedBackends` so the client can show ML AprilTag controls only when Orin acceleration is available.

The preferred runtime model flow is:

1. Ship or install an AprilTag YOLO model source with the Orin image.
2. On Orin, build or load a TensorRT-compatible cached artifact for that exact JetPack/TensorRT runtime.
3. Load the cached artifact through the CUDA detector.
4. Fall back to traditional AprilTag detection if model load or inference fails.

This avoids using Rubik TFLite or RKNN-specific model formats on Orin.

## AprilTag Pipeline Flow

The normal AprilTag pipeline will gain optional ML-assisted detection settings:

- Enable ML detection.
- ML model name.
- Confidence threshold.
- NMS threshold.
- ROI padding in pixels.
- Fallback to traditional detection.
- Show ROI boxes.

When ML detection is enabled and available:

1. Run the CUDA/TensorRT object detector on the color frame.
2. Convert detections into AprilTag ROIs.
3. Expand each ROI by configured pixel padding and clamp to frame bounds.
4. Decode AprilTags inside those ROIs with WPILib AprilTag detection.
5. Transform decoded corners and homography back into full-frame coordinates.
6. Deduplicate tags by ID, keeping the best decision margin.
7. Run existing single-tag and multi-tag pose estimation unchanged.
8. Draw ROI boxes on processed output when enabled.

When ML detection is disabled or unavailable:

1. Run the current full-frame AprilTag detector.
2. Run pose estimation unchanged.

When ML detection finds no tags:

1. If fallback is enabled, run full-frame AprilTag detection.
2. If fallback is disabled, return no targets.

## UI

Add ML AprilTag controls to the existing AprilTag tab. The controls should appear only when `supportedBackends` contains the Orin CUDA/TensorRT backend.

The UI should say CUDA or GPU instead of NPU on this branch.

The controls should follow the DoctorFogarty ML branch behavior but avoid unrelated UI changes.

## Error Handling

- Missing CUDA/TensorRT native library: log once, mark backend unavailable, keep traditional AprilTag path working.
- Missing model: show no ML model choices, keep traditional AprilTag path working.
- TensorRT model build/load failure: log with model path and platform details, keep traditional AprilTag path working.
- Empty ROI set: optionally fall back to traditional detection according to pipeline settings.
- Bad ROI coordinates: clamp and skip invalid ROIs.
- NetworkManager unavailable in image: PhotonVision should log network management failure but still start the web service.

## Testing

Add or adapt focused tests:

- `NetworkConfigTest` for Team 2375 Orin defaults.
- `NetworkManagerTest` for selecting a wired interface before an active connection exists.
- `VisionModuleCameraSettingsTest` for exposure-before-auto-exposure ordering.
- ROI expansion and clamping tests.
- ROI decode coordinate/homography mapping tests.
- ML pipeline flag safety test: disabled or unavailable ML must not change traditional AprilTag behavior.
- Backend selection test: Jetson/Orin exposes CUDA/TensorRT as a supported backend.

Run focused Gradle tests before implementation is considered complete:

- `./gradlew.bat photon-core:test --tests org.photonvision.common.configuration.NetworkConfigTest`
- `./gradlew.bat photon-core:test --tests org.photonvision.common.networking.NetworkManagerTest`
- `./gradlew.bat photon-core:test --tests org.photonvision.vision.processes.VisionModuleCameraSettingsTest`
- ROI and ML AprilTag tests added by this work.

Image verification:

- Build the `LinuxArm64` JAR.
- Build the Orin Nano image artifact.
- Boot on Orin Nano.
- Confirm web UI at `http://10.23.75.15:5800`.
- Confirm hostname `photonvision-orin-nano-ml-tag`.
- Confirm no Wi-Fi fallback profile is installed.
- Confirm manual exposure applies after boot without UI toggling.
- Confirm ML AprilTag controls appear.
- Confirm ML AprilTag detection can be enabled and pose estimation still works.

## Rollout

Commit in logical slices:

1. Orin/static network and image design/spec.
2. Network defaults and image runner changes.
3. Exposure startup fix.
4. Jetson/Orin platform detection.
5. CUDA/TensorRT model backend.
6. ML AprilTag ROI pipeline.
7. UI controls and settings.
8. Tests and final verification fixes.

## Risks

- TensorRT engine compatibility is tied to JetPack/TensorRT versions. The image should build or cache the runtime artifact on Orin, or pin the base image and engine together.
- The 4143 CUDA AprilTag branch loads an external `971apriltag` library that is not present in this repo. This work should not depend on that library unless the native artifact is added and packaged explicitly.
- Orin base image root partition layout may differ from existing PhotonVision images. The workflow must pin `root_location` and smoke-test the image.
- First boot may take longer if a TensorRT artifact is generated on device.
- Global networking defaults mean all artifacts from `image/orin-nano` default to Team 2375 static networking, by request.
