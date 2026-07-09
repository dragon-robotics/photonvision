# Orin Nano ML Tag Image Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Team 2375 Orin Nano PhotonVision image with static networking, fixed manual exposure startup, and ML-assisted AprilTag ROI detection using `models/source/apriltag/best.pt` as the source model.

**Architecture:** Keep PhotonVision `v2026.3.4` and Java 17 intact. Export `best.pt` to ONNX for packaging, generate/cache TensorRT engines on the Orin runtime, and route Orin inference through the existing `Model` and `ObjectDetector` abstraction so the normal AprilTag pipeline can reuse the ML ROI flow.

**Tech Stack:** Java 17, Gradle, PhotonVision `linuxarm64`, WPILib AprilTag, OpenCV, YOLO `.pt`, ONNX, TensorRT/CUDA on NVIDIA Jetson Orin Nano, NetworkManager.

## Global Constraints

- Branch is `image/orin-nano`.
- Base PhotonVision version is `v2026.3.4`.
- Java stays at 17.
- Team number default is `2375`.
- Static IP default is `10.23.75.15`.
- Gateway is `10.23.75.4`.
- Hostname default is `photonvision-orin-nano-ml-tag`.
- No Wi-Fi fallback profile is added.
- Do not merge the full `FRC-Team-4143/photonvision` `jetson-orin` branch.
- Do not merge the full `DoctorFogarty/photonvision` `apriltag-ml-experimental` branch.
- Do not use Rubik TFLite delegate or RKNN for Orin inference.
- Source model path is `models/source/apriltag/best.pt`.
- Runtime ONNX model path is `photon-server/src/main/resources/models/apriltag-640-640-yolo.onnx`.
- TensorRT `.engine` files are generated or cached on the Orin runtime, not committed.

---

## File Structure

- `models/source/apriltag/best.pt`: committed source YOLO model for AprilTag ROI detection.
- `models/source/apriltag/README.md`: records model source and export command.
- `photon-server/src/main/resources/models/apriltag-640-640-yolo.onnx`: generated runtime model shipped in the JAR.
- `photon-core/src/main/java/org/photonvision/common/configuration/NetworkConfig.java`: global Team 2375 Orin network defaults.
- `photon-core/src/main/java/org/photonvision/common/networking/NetworkManager.java`: early wired-interface selection before NetworkManager reports an active connection.
- `photon-core/src/main/java/org/photonvision/vision/processes/VisionModule.java`: exposure-before-auto-exposure camera settings ordering.
- `photon-targeting/src/main/java/org/photonvision/common/hardware/Platform.java`: Jetson/Orin platform detection.
- `photon-core/src/main/java/org/photonvision/common/configuration/NeuralNetworkModelManager.java`: `TENSORRT` backend family and shipped AprilTag model registration.
- `photon-core/src/main/java/org/photonvision/common/configuration/NeuralNetworkModelsSettings.java`: model filename parsing and family support for ONNX/TensorRT models.
- `photon-core/src/main/java/org/photonvision/vision/objects/TensorRtModel.java`: model wrapper for ONNX models used by TensorRT.
- `photon-core/src/main/java/org/photonvision/vision/objects/TensorRtObjectDetector.java`: Java object detector implementation for Orin TensorRT inference.
- `photon-core/src/main/java/org/photonvision/vision/objects/TensorRtJNI.java`: JNI facade for native TensorRT detector calls.
- `photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagROIDetectionPipe.java`: ROI detection pipe using `ObjectDetector`.
- `photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagROIDecodePipe.java`: ROI decode and homography/corner remapping.
- `photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagMLHybridPipe.java`: combines ML ROI detection and ROI decode into one testable unit.
- `photon-core/src/main/java/org/photonvision/vision/pipe/impl/MLDetectionResult.java`: result object carrying detections, ROIs, and elapsed time.
- `photon-core/src/main/java/org/photonvision/vision/pipeline/AprilTagPipeline.java`: optional ML-assisted branch.
- `photon-core/src/main/java/org/photonvision/vision/pipeline/AprilTagPipelineSettings.java`: ML settings.
- `photon-core/src/main/java/org/photonvision/vision/pipeline/OutputStreamPipeline.java`: ROI box drawing.
- `photon-core/src/main/java/org/photonvision/vision/pipeline/result/CVPipelineResult.java`: result field for ML ROIs.
- `photon-client/src/types/PipelineTypes.ts`: client ML AprilTag settings.
- `photon-client/src/types/SettingTypes.ts`: client model family type includes `TENSORRT`.
- `photon-client/src/components/dashboard/tabs/AprilTagTab.vue`: ML AprilTag controls.
- `scripts/armrunner.sh`: install JAR and write static Orin NetworkManager profile.
- `.github/workflows/build.yml`: Orin Nano image matrix entry using repository variable `ORIN_NANO_IMAGE_URL`.

---

### Task 1: Commit Source Model and Generate ONNX

**Files:**
- Keep: `models/source/apriltag/best.pt`
- Keep: `models/source/apriltag/README.md`
- Create: `photon-server/src/main/resources/models/apriltag-640-640-yolo.onnx`

**Interfaces:**
- Consumes: `models/source/apriltag/best.pt`
- Produces: `photon-server/src/main/resources/models/apriltag-640-640-yolo.onnx`

- [ ] **Step 1: Verify source model exists**

Run:

```powershell
Get-Item -LiteralPath 'models/source/apriltag/best.pt' | Select-Object FullName, Length
```

Expected: `Length` is `5493594`.

- [ ] **Step 2: Export ONNX from the source model**

Run from repo root in an environment with Ultralytics installed:

```powershell
yolo export model=models/source/apriltag/best.pt format=onnx imgsz=640 simplify=True opset=12
```

Expected: export succeeds and writes an ONNX file next to `best.pt`.

- [ ] **Step 3: Move ONNX into PhotonVision runtime resources**

Run:

```powershell
Move-Item -LiteralPath 'models/source/apriltag/best.onnx' -Destination 'photon-server/src/main/resources/models/apriltag-640-640-yolo.onnx' -Force
```

Expected: `photon-server/src/main/resources/models/apriltag-640-640-yolo.onnx` exists.

- [ ] **Step 4: Validate ONNX file presence**

Run:

```powershell
Get-Item -LiteralPath 'photon-server/src/main/resources/models/apriltag-640-640-yolo.onnx' | Select-Object FullName, Length
```

Expected: file exists and has nonzero `Length`.

- [ ] **Step 5: Commit source model and ONNX artifact**

Run:

```powershell
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' add models/source/apriltag/best.pt models/source/apriltag/README.md photon-server/src/main/resources/models/apriltag-640-640-yolo.onnx
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' commit -m "Add Orin AprilTag ROI source model"
```

Expected: commit contains exactly the source model, README, and ONNX runtime model.

---

### Task 2: Team 2375 Network Defaults and Image Network Profile

**Files:**
- Modify: `photon-core/src/main/java/org/photonvision/common/configuration/NetworkConfig.java`
- Modify: `photon-core/src/main/java/org/photonvision/common/networking/NetworkManager.java`
- Modify: `scripts/armrunner.sh`
- Test: `photon-core/src/test/java/org/photonvision/common/configuration/NetworkConfigTest.java`
- Test: `photon-core/src/test/java/org/photonvision/common/networking/NetworkManagerTest.java`

**Interfaces:**
- Produces: global defaults `2375`, `STATIC`, `10.23.75.15`, `photonvision-orin-nano-ml-tag`.
- Produces: `NetworkManager.selectManagedInterface(NetworkConfig, List<NMDeviceInfo>)`.

- [ ] **Step 1: Add failing NetworkConfig defaults test**

Add test method:

```java
@Test
public void testTeam2375OrinDefaults() {
    var config = new NetworkConfig();

    assertEquals("2375", config.ntServerAddress);
    assertEquals(NetworkMode.STATIC, config.connectionType);
    assertEquals("10.23.75.15", config.staticIp);
    assertEquals("photonvision-orin-nano-ml-tag", config.hostname);
}
```

Run:

```powershell
./gradlew.bat photon-core:test --tests org.photonvision.common.configuration.NetworkConfigTest
```

Expected: fails because defaults still use DHCP/`photonvision`.

- [ ] **Step 2: Change NetworkConfig defaults**

Set:

```java
public String ntServerAddress = "2375";
public NetworkMode connectionType = NetworkMode.STATIC;
public String staticIp = "10.23.75.15";
public String hostname = "photonvision-orin-nano-ml-tag";
```

Run the same test. Expected: pass.

- [ ] **Step 3: Add wired-interface selection test**

Add test:

```java
@Test
public void selectsWiredInterfaceBeforeNetworkManagerConnectionIsActive() {
    var config = new NetworkConfig();
    config.networkManagerIface = "";

    var selected =
            NetworkManager.selectManagedInterface(
                    config, List.of(new NMDeviceInfo("", "enxf074e47f4b5b", "ethernet")));

    assertEquals("enxf074e47f4b5b", selected);
}
```

Run:

```powershell
./gradlew.bat photon-core:test --tests org.photonvision.common.networking.NetworkManagerTest
```

Expected: fails because `selectManagedInterface` does not exist.

- [ ] **Step 4: Backport `selectManagedInterface` from `2026-team2375`**

Add:

```java
static String selectManagedInterface(NetworkConfig config, List<NMDeviceInfo> ethernetDevices) {
    if (config.networkManagerIface != null && !config.networkManagerIface.isBlank()) {
        var configuredDevice =
                ethernetDevices.stream()
                        .filter(it -> it.devName().equals(config.networkManagerIface))
                        .findFirst();
        if (configuredDevice.isPresent() || ethernetDevices.isEmpty()) {
            return config.networkManagerIface;
        }
    }

    return ethernetDevices.stream()
            .filter(it -> !it.connName().isBlank())
            .findFirst()
            .or(() -> ethernetDevices.stream().findFirst())
            .map(NMDeviceInfo::devName)
            .orElse(config.networkManagerIface);
}
```

Wire `initialize()` to use this helper with `NetworkUtils.getAllWiredInterfaces()`. Run both network tests. Expected: pass.

- [ ] **Step 5: Add Orin static NetworkManager profile to `scripts/armrunner.sh`**

Add a `static-team2375-orin.nmconnection` block:

```ini
[connection]
id=static-team2375-orin
uuid=23752375-2375-4375-8375-000000000015
type=ethernet
autoconnect=true
autoconnect-priority=100

[ethernet]

[ipv4]
method=manual
address1=10.23.75.15/8,10.23.75.4
may-fail=false

[ipv6]
method=disabled
```

Also remove stale config:

```bash
sudo rm -f ${DEST_PV_LOCATION}/photonvision_config/photon.sqlite*
sudo rm -f ${DEST_PV_LOCATION}/photonvision_config/networkSettings.json
```

Do not add a Wi-Fi profile.

- [ ] **Step 6: Run tests and commit**

Run:

```powershell
./gradlew.bat photon-core:test --tests org.photonvision.common.configuration.NetworkConfigTest --tests org.photonvision.common.networking.NetworkManagerTest
```

Expected: both test classes pass.

Commit:

```powershell
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' add photon-core/src/main/java/org/photonvision/common/configuration/NetworkConfig.java photon-core/src/main/java/org/photonvision/common/networking/NetworkManager.java photon-core/src/test/java/org/photonvision/common/configuration/NetworkConfigTest.java photon-core/src/test/java/org/photonvision/common/networking/NetworkManagerTest.java scripts/armrunner.sh
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' commit -m "Configure Orin Nano team network defaults"
```

---

### Task 3: Manual Exposure Startup Fix

**Files:**
- Modify: `photon-core/src/main/java/org/photonvision/vision/processes/VisionModule.java`
- Test: `photon-core/src/test/java/org/photonvision/vision/processes/VisionModuleCameraSettingsTest.java`

**Interfaces:**
- Produces: `VisionModule.applyCameraSettingsForPipeline(VisionSourceSettables, CVPipelineSettings, QuirkyCamera)`.

- [ ] **Step 1: Add failing camera settings order test**

Add `VisionModuleCameraSettingsTest` from `2026-team2375`, including:

```java
@Test
void manualPipelinePrimesExposureBeforeDisablingAutoExposure() {
    var settables = new RecordingSettables();
    var settings = new CVPipelineSettings();
    settings.cameraAutoExposure = false;
    settings.cameraExposureRaw = 37;

    VisionModule.applyCameraSettingsForPipeline(settables, settings, QuirkyCamera.DefaultCamera);

    assertTrue(
            settables.calls.indexOf("exposure:37.0") < settables.calls.indexOf("autoExposure:false"));
}
```

Run:

```powershell
./gradlew.bat photon-core:test --tests org.photonvision.vision.processes.VisionModuleCameraSettingsTest
```

Expected: compile fails because helper method is not present.

- [ ] **Step 2: Extract and reorder camera settings application**

In `VisionModule.setPipeline`, replace inline camera setup with:

```java
applyCameraSettingsForPipeline(settables, pipelineSettings, cameraQuirks);
```

Add helper:

```java
static void applyCameraSettingsForPipeline(
        VisionSourceSettables settables,
        CVPipelineSettings pipelineSettings,
        QuirkyCamera cameraQuirks) {
    settables.setVideoModeInternal(pipelineSettings.cameraVideoModeIndex);
    settables.setBrightness(pipelineSettings.cameraBrightness);

    if (!pipelineSettings.cameraAutoExposure) {
        if (pipelineSettings.cameraExposureRaw < 0) pipelineSettings.cameraExposureRaw = 10;
        settables.setExposureRaw(pipelineSettings.cameraExposureRaw);
    }

    try {
        settables.setAutoExposure(pipelineSettings.cameraAutoExposure);
    } catch (VideoException e) {
        settables.logger.error("Unable to set camera auto exposure!");
        settables.logger.error(e.toString());
    }

    if (cameraQuirks.hasQuirk(CameraQuirk.Gain)) {
        if (pipelineSettings.cameraGain == -1) pipelineSettings.cameraGain = 75;
        settables.setGain(Math.max(0, pipelineSettings.cameraGain));
    } else {
        pipelineSettings.cameraGain = -1;
    }

    if (cameraQuirks.hasQuirk(CameraQuirk.AwbRedBlueGain)) {
        if (pipelineSettings.cameraRedGain == -1) pipelineSettings.cameraRedGain = 11;
        if (pipelineSettings.cameraBlueGain == -1) pipelineSettings.cameraBlueGain = 20;
        settables.setRedGain(Math.max(0, pipelineSettings.cameraRedGain));
        settables.setBlueGain(Math.max(0, pipelineSettings.cameraBlueGain));
    } else {
        pipelineSettings.cameraRedGain = -1;
        pipelineSettings.cameraBlueGain = -1;
        settables.setWhiteBalanceTemp(pipelineSettings.cameraWhiteBalanceTemp);
        settables.setAutoWhiteBalance(pipelineSettings.cameraAutoWhiteBalance);
    }
}
```

- [ ] **Step 3: Run test and commit**

Run:

```powershell
./gradlew.bat photon-core:test --tests org.photonvision.vision.processes.VisionModuleCameraSettingsTest
```

Expected: pass.

Commit:

```powershell
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' add photon-core/src/main/java/org/photonvision/vision/processes/VisionModule.java photon-core/src/test/java/org/photonvision/vision/processes/VisionModuleCameraSettingsTest.java
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' commit -m "Fix manual exposure startup on Orin"
```

---

### Task 4: Jetson/Orin Platform and TensorRT Model Family

**Files:**
- Modify: `photon-targeting/src/main/java/org/photonvision/common/hardware/Platform.java`
- Modify: `photon-core/src/main/java/org/photonvision/common/configuration/NeuralNetworkModelManager.java`
- Modify: `photon-core/src/main/java/org/photonvision/common/configuration/NeuralNetworkModelsSettings.java`
- Test: `photon-core/src/test/java/org/photonvision/common/configuration/NeuralNetworkModelManagerTest.java`

**Interfaces:**
- Produces: `Platform.isJetson()`.
- Produces: `Family.TENSORRT` with extension `.onnx`.

- [ ] **Step 1: Add platform helper**

Add to `Platform`:

```java
public static boolean isJetson() {
    return currentPlatform == LINUX_AARCH64 && fileHasText("/proc/device-tree/model", "NVIDIA Jetson");
}
```

Ensure Jetson AArch64 detection returns `LINUX_AARCH64`.

- [ ] **Step 2: Add TensorRT family**

Change enum:

```java
public enum Family {
    RKNN(".rknn"),
    RUBIK(".tflite"),
    TENSORRT(".onnx");
}
```

- [ ] **Step 3: Register Orin backend**

In `NeuralNetworkModelManager` constructor, add:

```java
case LINUX_AARCH64 -> {
    if (Platform.isJetson()) supportedBackends.add(Family.TENSORRT);
}
```

- [ ] **Step 4: Register shipped AprilTag ONNX model**

In `getShippedProperties`, add:

```java
nnProps.addModelProperties(
        new ModelProperties(
                Path.of(modelsDirectory.getAbsolutePath(), "apriltag-640-640-yolo.onnx"),
                "AprilTag ROI",
                new LinkedList<String>(List.of("AprilTag")),
                640,
                640,
                Family.TENSORRT,
                Version.YOLOV8));
```

- [ ] **Step 5: Add tests and run**

Add tests that verify:

```java
assertEquals(".onnx", Family.TENSORRT.extension());
```

Run:

```powershell
./gradlew.bat photon-core:test --tests org.photonvision.common.configuration.NeuralNetworkModelManagerTest
```

Expected: pass.

- [ ] **Step 6: Commit**

```powershell
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' add photon-targeting/src/main/java/org/photonvision/common/hardware/Platform.java photon-core/src/main/java/org/photonvision/common/configuration/NeuralNetworkModelManager.java photon-core/src/main/java/org/photonvision/common/configuration/NeuralNetworkModelsSettings.java photon-core/src/test/java/org/photonvision/common/configuration/NeuralNetworkModelManagerTest.java
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' commit -m "Add Orin TensorRT model backend"
```

---

### Task 5: TensorRT Java Detector Facade

**Files:**
- Create: `photon-core/src/main/java/org/photonvision/vision/objects/TensorRtJNI.java`
- Create: `photon-core/src/main/java/org/photonvision/vision/objects/TensorRtModel.java`
- Create: `photon-core/src/main/java/org/photonvision/vision/objects/TensorRtObjectDetector.java`
- Modify: `photon-core/src/main/java/org/photonvision/common/configuration/NeuralNetworkModelManager.java`
- Test: `photon-core/src/test/java/org/photonvision/common/configuration/NeuralNetworkModelManagerTest.java`

**Interfaces:**
- Produces: `TensorRtModel.load(): ObjectDetector`.
- Produces: `TensorRtObjectDetector.detect(Mat, double, double): List<NeuralNetworkPipeResult>`.
- Produces: native facade signatures `create`, `detect`, `destroy`.

- [ ] **Step 1: Add `TensorRtJNI` facade**

Create:

```java
package org.photonvision.vision.objects;

import org.opencv.core.Rect2d;

public final class TensorRtJNI {
    private TensorRtJNI() {}

    public record Detection(Rect2d rect, int classId, double confidence) {}

    public static boolean isAvailable() {
        try {
            System.loadLibrary("photon_tensorrt_jni");
            return true;
        } catch (UnsatisfiedLinkError error) {
            return false;
        }
    }

    public static native long create(String onnxPath, String engineCachePath, int width, int height, int classCount);

    public static native Detection[] detect(long ptr, long matAddr, double nmsThresh, double boxThresh);

    public static native void destroy(long ptr);
}
```

- [ ] **Step 2: Add `TensorRtModel`**

Create a model class matching `RubikModel` shape, requiring `Family.TENSORRT` and `Version.YOLOV8` or `Version.YOLOV11`.

- [ ] **Step 3: Add `TensorRtObjectDetector`**

Create detector that:

```java
Mat letterboxed = new Mat();
Letterbox scale =
        Letterbox.letterbox(in, letterboxed, this.inputSize, ColorHelper.colorToScalar(Color.GRAY));
var results = TensorRtJNI.detect(ptr, letterboxed.getNativeObjAddr(), nmsThresh, boxThresh);
letterboxed.release();
return scale.resizeDetections(List.of(results).stream()
        .map(it -> new NeuralNetworkPipeResult(it.rect(), it.classId(), it.confidence()))
        .toList());
```

If `TensorRtJNI.isAvailable()` is false or `create()` returns `0`, throw `UnsupportedOperationException` with model filename.

- [ ] **Step 4: Wire model manager**

In `loadModel` switch:

```java
case TENSORRT -> {
    models.get(properties.family()).add(new TensorRtModel(properties));
}
```

- [ ] **Step 5: Run compile test**

Run:

```powershell
./gradlew.bat photon-core:testClasses
```

Expected: Java compiles.

- [ ] **Step 6: Commit**

```powershell
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' add photon-core/src/main/java/org/photonvision/vision/objects/TensorRtJNI.java photon-core/src/main/java/org/photonvision/vision/objects/TensorRtModel.java photon-core/src/main/java/org/photonvision/vision/objects/TensorRtObjectDetector.java photon-core/src/main/java/org/photonvision/common/configuration/NeuralNetworkModelManager.java
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' commit -m "Add TensorRT detector facade"
```

---

### Task 6: ML AprilTag ROI Pipeline

**Files:**
- Create: `photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagROIDetectionPipe.java`
- Create: `photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagROIDecodePipe.java`
- Create: `photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagMLHybridPipe.java`
- Create: `photon-core/src/main/java/org/photonvision/vision/pipe/impl/MLDetectionResult.java`
- Modify: `photon-core/src/main/java/org/photonvision/vision/pipeline/AprilTagPipeline.java`
- Modify: `photon-core/src/main/java/org/photonvision/vision/pipeline/AprilTagPipelineSettings.java`
- Test: `photon-core/src/test/java/org/photonvision/vision/pipe/AprilTagROIDecodePipeTest.java`
- Test: `photon-core/src/test/java/org/photonvision/vision/pipe/AprilTagMLHybridPipeTest.java`

**Interfaces:**
- Consumes: `ObjectDetector.detect(Mat, double, double)`.
- Produces: `MLDetectionResult(List<AprilTagDetection> detections, List<Rect2d> rois, long nanosElapsed)`.

- [ ] **Step 1: Backport ROI detection pipe from `2026-team2375`**

Use the local `2026-team2375` implementation of `AprilTagROIDetectionPipe`, preserving `ObjectDetector` abstraction and returning `List<Rect2d>`.

- [ ] **Step 2: Backport ROI decode pipe tests**

Use `AprilTagROIDecodePipeTest` from `2026-team2375`, including tests for:

- expanding ROI by additive pixels
- clamping ROI to frame bounds
- preserving full-frame mapped corners
- deduplicating by tag ID

Run:

```powershell
./gradlew.bat photon-core:test --tests org.photonvision.vision.pipe.AprilTagROIDecodePipeTest
```

Expected: pass after pipe is added.

- [ ] **Step 3: Add ML settings**

Add fields to `AprilTagPipelineSettings`:

```java
public boolean useMLDetection = false;
public double mlConfidenceThreshold = 0.5;
public double mlNmsThreshold = 0.45;
public int mlRoiPaddingPixels = 40;
public boolean mlFallbackToTraditional = true;
public String mlModelName = null;
public boolean showDetectionBoxes = true;
```

Update `equals` and `hashCode` for these fields.

- [ ] **Step 4: Wire AprilTagPipeline hybrid path**

Use this behavior:

```java
if (settings.useMLDetection && mlHybridPipe.isAvailable()) {
    var mlResult = mlHybridPipe.run(frame);
    detections = mlResult.output.detections();
    frame.mlDetectionRois = mlResult.output.rois();
    if (detections.isEmpty() && settings.mlFallbackToTraditional) {
        var fallbackResult = aprilTagDetectionPipe.run(frame.processedImage);
        detections = fallbackResult.output;
    }
} else {
    var tagDetectionPipeResult = aprilTagDetectionPipe.run(frame.processedImage);
    detections = tagDetectionPipeResult.output;
}
```

- [ ] **Step 5: Run ML safety tests**

Run:

```powershell
./gradlew.bat photon-core:test --tests org.photonvision.vision.pipe.AprilTagMLHybridPipeTest
```

Expected: disabled or unavailable ML returns traditional behavior or empty ML result without crash.

- [ ] **Step 6: Commit**

```powershell
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' add photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagROIDetectionPipe.java photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagROIDecodePipe.java photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagMLHybridPipe.java photon-core/src/main/java/org/photonvision/vision/pipe/impl/MLDetectionResult.java photon-core/src/main/java/org/photonvision/vision/pipeline/AprilTagPipeline.java photon-core/src/main/java/org/photonvision/vision/pipeline/AprilTagPipelineSettings.java photon-core/src/test/java/org/photonvision/vision/pipe/AprilTagROIDecodePipeTest.java photon-core/src/test/java/org/photonvision/vision/pipe/AprilTagMLHybridPipeTest.java
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' commit -m "Add Orin ML AprilTag ROI pipeline"
```

---

### Task 7: UI Controls and ROI Stream Overlay

**Files:**
- Modify: `photon-client/src/types/PipelineTypes.ts`
- Modify: `photon-client/src/types/SettingTypes.ts`
- Modify: `photon-client/src/components/dashboard/tabs/AprilTagTab.vue`
- Create: `photon-core/src/main/java/org/photonvision/vision/pipe/impl/DrawMLROIPipe.java`
- Modify: `photon-core/src/main/java/org/photonvision/vision/pipeline/OutputStreamPipeline.java`
- Modify: `photon-core/src/main/java/org/photonvision/vision/pipeline/result/CVPipelineResult.java`

**Interfaces:**
- Consumes: `supportedBackends` containing `TENSORRT`.
- Produces: AprilTag tab ML controls labeled for CUDA/GPU.
- Produces: optional ROI boxes on processed stream.

- [ ] **Step 1: Add client setting types**

Add `TENSORRT` to model family type:

```ts
family: "RKNN" | "RUBIK" | "TENSORRT";
```

Add AprilTag ML settings to `AprilTagPipelineSettings` and defaults matching Java.

- [ ] **Step 2: Add AprilTag UI controls**

In `AprilTagTab.vue`, display ML controls when:

```ts
const mlDetectionAvailable = computed(() => useSettingsStore().general.supportedBackends.includes("TENSORRT"));
```

Use label text `AI-Assisted Detection (CUDA)` and tooltip text that references Orin GPU acceleration.

- [ ] **Step 3: Add ROI drawing pipe**

Backport `DrawMLROIPipe` from `2026-team2375` and draw boxes only when `showDetectionBoxes` is true.

- [ ] **Step 4: Run compile checks**

Run:

```powershell
./gradlew.bat photon-core:testClasses
```

Run:

```powershell
Set-Location photon-client
pnpm typecheck
```

Expected: Java test classes compile and client types compile.

- [ ] **Step 5: Commit**

```powershell
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' add photon-client/src/types/PipelineTypes.ts photon-client/src/types/SettingTypes.ts photon-client/src/components/dashboard/tabs/AprilTagTab.vue photon-core/src/main/java/org/photonvision/vision/pipe/impl/DrawMLROIPipe.java photon-core/src/main/java/org/photonvision/vision/pipeline/OutputStreamPipeline.java photon-core/src/main/java/org/photonvision/vision/pipeline/result/CVPipelineResult.java
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' commit -m "Add Orin ML AprilTag controls"
```

---

### Task 8: Orin Nano Image Build Workflow

**Files:**
- Modify: `.github/workflows/build.yml`
- Modify: `scripts/armrunner.sh`

**Interfaces:**
- Produces: GitHub Actions artifact `image-orinnano`.
- Produces: image filename ending `_orinnano.img.xz`.
- Consumes: GitHub Actions repository variable `ORIN_NANO_IMAGE_URL`, containing a pinned `.img.xz` Orin Nano base image URL.

- [ ] **Step 1: Add image matrix row**

Add matrix entry:

```yaml
- os: ubuntu-24.04-arm
  artifact-name: LinuxArm64
  image_suffix: orinnano
  plat_override: LINUX_AARCH64
  image_url: ${{ vars.ORIN_NANO_IMAGE_URL }}
  minimum_free_mb: 2048
```

Set GitHub Actions repository variable `ORIN_NANO_IMAGE_URL` to the pinned Orin Nano base image release URL before running the image job. Keep existing Rubik entries unchanged.

- [ ] **Step 2: Add workflow guard for missing Orin image URL**

Add a step before `photonvision/photon-image-runner@HEAD`:

```yaml
- name: Validate Orin image URL
  if: ${{ matrix.image_suffix == 'orinnano' && matrix.image_url == '' }}
  run: |
    echo "ORIN_NANO_IMAGE_URL repository variable must point to a pinned Orin Nano .img.xz base image"
    exit 1
```

- [ ] **Step 3: Keep standard image compression path**

The existing non-Rubik compress step should produce:

```text
photonvision-..._orinnano.img.xz
```

- [ ] **Step 4: Add smoketest platform override**

Confirm smoketest command uses:

```bash
java -jar *.jar --smoketest --platform=LINUX_AARCH64
```

- [ ] **Step 5: Run workflow syntax check**

Run:

```powershell
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' diff --check -- .github/workflows/build.yml scripts/armrunner.sh
```

Expected: no whitespace errors.

- [ ] **Step 6: Commit**

```powershell
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' add .github/workflows/build.yml scripts/armrunner.sh
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' commit -m "Build Orin Nano PhotonVision image"
```

---

### Task 9: Verification on Host and Orin Nano

**Files:**
- No planned source edits.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: verified JAR, verified Orin image, hardware notes.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
./gradlew.bat photon-core:test --tests org.photonvision.common.configuration.NetworkConfigTest --tests org.photonvision.common.networking.NetworkManagerTest --tests org.photonvision.vision.processes.VisionModuleCameraSettingsTest --tests org.photonvision.vision.pipe.AprilTagROIDecodePipeTest --tests org.photonvision.vision.pipe.AprilTagMLHybridPipeTest
```

Expected: all targeted tests pass.

- [ ] **Step 2: Build LinuxArm64 JAR**

Run:

```powershell
./gradlew.bat photon-targeting:jar photon-server:shadowJar -PArchOverride=linuxarm64
```

Expected: `photon-server/build/libs/*linuxarm64.jar` exists.

- [ ] **Step 3: Build Orin image**

Run the GitHub Actions image workflow or local image runner path selected for the Orin base image.

Expected: `_orinnano.img.xz` artifact exists.

- [ ] **Step 4: Flash and boot Orin Nano**

Flash image to Orin Nano media, boot, then verify:

```bash
hostname
ip addr
nmcli connection show
systemctl status photonvision
```

Expected:

- hostname is `photonvision-orin-nano-ml-tag`
- Ethernet has `10.23.75.15/8`
- default gateway is `10.23.75.4`
- no Wi-Fi fallback connection exists
- PhotonVision service is active

- [ ] **Step 5: Verify web UI**

Open:

```text
http://10.23.75.15:5800
```

Expected: UI loads, AprilTag tab shows CUDA ML controls when Orin backend is available.

- [ ] **Step 6: Verify exposure**

Reboot Orin Nano with a manual-exposure AprilTag pipeline selected.

Expected: manual exposure applies after boot without toggling auto exposure in the UI.

- [ ] **Step 7: Verify ML AprilTag pose**

Enable ML AprilTag detection and view targets.

Expected:

- ROI boxes appear when enabled.
- Tags decode.
- MultiTag pose estimation still produces field-to-camera output.
- Traditional fallback works when ML is disabled.

---

## Execution Order

1. Task 1: source model and ONNX.
2. Task 2: network defaults and static image profile.
3. Task 3: manual exposure startup fix.
4. Task 4: Jetson/Orin platform and TensorRT family.
5. Task 5: TensorRT Java detector facade.
6. Task 6: ML AprilTag ROI pipeline.
7. Task 7: UI controls and ROI overlay.
8. Task 8: image workflow.
9. Task 9: host and Orin Nano verification.

## Current Working Tree Notes

At plan creation time, these files are intentionally untracked:

```text
models/source/apriltag/best.pt
models/source/apriltag/README.md
```

This unrelated file is also untracked and should not be staged by this work:

```text
photon-client/pnpm-workspace.yaml
```
