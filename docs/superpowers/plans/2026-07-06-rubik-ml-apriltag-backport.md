# Rubik ML AprilTag Backport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backport DoctorFogarty's ML-assisted AprilTag ROI detection to `2026-team2375` for Rubik Pi 3 only.

**Architecture:** Add a Rubik-only AprilTag ML model and route AprilTag detection through a guarded ML ROI path when enabled. The ML path finds ROIs with the Rubik object detector, decodes AprilTags inside those ROIs with the existing WPILib detector, maps detections back into full-frame coordinates, then feeds the existing single-tag and multi-tag pose code. Traditional full-frame AprilTag detection remains the default and the runtime fallback.

**Tech Stack:** Java 17, Gradle 8.14.3, PhotonVision `v2026.3.4`, WPILib 2026 `edu.wpi.first.*` packages, OpenCV, Rubik `.tflite` object detector, Vue 3 + Pinia + Vuetify.

---

## File Structure

- Create `photon-core/src/test/java/org/photonvision/common/configuration/NeuralNetworkModelManagerTest.java` to prove AprilTag model metadata and lookup behavior.
- Modify `photon-core/src/main/java/org/photonvision/common/configuration/NeuralNetworkModelManager.java` to add Rubik AprilTag model metadata and an AprilTag-specific default lookup.
- Add `photon-server/src/main/resources/models/apriltagV4-yolo11.tflite` from `doctorfogarty/apriltag-ml-experimental-sync`.
- Create `photon-core/src/main/java/org/photonvision/vision/pipe/impl/MLDetectionResult.java` as a result record for ML detections and ROI rectangles.
- Create `photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagROIDetectionPipe.java` to run the Rubik object detector and return candidate ROI rectangles.
- Create `photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagROIDecodePipe.java` to decode AprilTags inside ROI rectangles and remap corners/homography to full-frame coordinates.
- Create `photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagMLHybridPipe.java` to compose ROI detection and ROI decode.
- Create `photon-core/src/main/java/org/photonvision/vision/pipe/impl/DrawMLROIPipe.java` to draw ROI boxes.
- Create `photon-core/src/test/java/org/photonvision/vision/pipe/AprilTagROIDecodePipeTest.java` to test ROI expansion, clamping, empty inputs, and coordinate mapping.
- Create `photon-core/src/test/java/org/photonvision/vision/pipe/AprilTagMLHybridPipeTest.java` to test host-only no-model behavior and release idempotence.
- Modify `photon-core/src/main/java/org/photonvision/vision/pipeline/AprilTagPipelineSettings.java` to add ML settings.
- Modify `photon-core/src/main/java/org/photonvision/vision/pipeline/AprilTagPipeline.java` to choose ML or traditional detection.
- Modify `photon-core/src/main/java/org/photonvision/vision/pipeline/result/CVPipelineResult.java` to carry ROI rectangles for visualization.
- Modify `photon-core/src/main/java/org/photonvision/vision/pipeline/OutputStreamPipeline.java` to draw ROI boxes.
- Modify `photon-core/src/main/java/org/photonvision/vision/processes/VisionModule.java` to pass ROI rectangles into the stream thread.
- Create `photon-core/src/test/java/org/photonvision/vision/pipeline/AprilTagPipelineMLFlagSafetyTest.java` to prove fallback and settings behavior.
- Modify `photon-client/src/types/PipelineTypes.ts` to add ML fields to AprilTag settings.
- Modify `photon-client/src/components/dashboard/tabs/AprilTagTab.vue` to show Rubik-only ML controls.
- Do not modify `scripts/armrunner.sh`, `.github/workflows/build.yml`, Java version files, Gradle wrapper, PhotonLib APIs, examples, website files, or docs outside this plan.

## Pre-Flight

- [ ] **Step 1: Confirm branch and dirty state**

Run:

```powershell
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' status -sb
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' branch --show-current
```

Expected:

```text
## 2026-team2375...origin/2026-team2375 [ahead 1]
?? photon-client/pnpm-workspace.yaml
2026-team2375
```

If `docs/superpowers/plans/2026-07-06-rubik-ml-apriltag-backport.md` is uncommitted, include it in a planning commit before implementation or leave it unstaged and do not include it in feature commits. Leave `photon-client/pnpm-workspace.yaml` untouched unless Task 4 explicitly decides the frontend build requires it.

- [ ] **Step 2: Ensure DoctorFogarty ref exists**

Run:

```powershell
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' fetch https://github.com/DoctorFogarty/photonvision.git apriltag-ml-experimental-sync:refs/remotes/doctorfogarty/apriltag-ml-experimental-sync
```

Expected: fetch completes with `doctorfogarty/apriltag-ml-experimental-sync` available.

---

### Task 1: Rubik AprilTag Model Metadata

**Files:**
- Modify: `photon-core/src/main/java/org/photonvision/common/configuration/NeuralNetworkModelManager.java`
- Create: `photon-core/src/test/java/org/photonvision/common/configuration/NeuralNetworkModelManagerTest.java`
- Add: `photon-server/src/main/resources/models/apriltagV4-yolo11.tflite`

- [ ] **Step 1: Write failing model manager tests**

Create `photon-core/src/test/java/org/photonvision/common/configuration/NeuralNetworkModelManagerTest.java`:

```java
package org.photonvision.common.configuration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Family;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Version;
import org.photonvision.common.configuration.NeuralNetworkModelsSettings.ModelProperties;
import org.photonvision.vision.objects.Model;
import org.photonvision.vision.objects.NullModel;
import org.photonvision.vision.objects.ObjectDetector;

public class NeuralNetworkModelManagerTest {
    @Test
    void shippedPropertiesIncludeRubikAprilTagModelOnly() throws Exception {
        var manager = NeuralNetworkModelManager.getInstance(true);
        var modelsDir = new File("build/test-models");

        Method method =
                NeuralNetworkModelManager.class.getDeclaredMethod("getShippedProperties", File.class);
        method.setAccessible(true);
        var settings = (NeuralNetworkModelsSettings) method.invoke(manager, modelsDir);

        var aprilTagModels =
                List.of(settings.getModels()).stream()
                        .filter(model -> model.nickname().equals("AprilTag V4"))
                        .toList();

        assertEquals(1, aprilTagModels.size());
        var aprilTagModel = aprilTagModels.get(0);
        assertEquals(Path.of(modelsDir.getAbsolutePath(), "apriltagV4-yolo11.tflite"), aprilTagModel.modelPath());
        assertEquals(List.of("AprilTag"), aprilTagModel.labels());
        assertEquals(640, aprilTagModel.resolutionWidth());
        assertEquals(640, aprilTagModel.resolutionHeight());
        assertEquals(Family.RUBIK, aprilTagModel.family());
        assertEquals(Version.YOLOV11, aprilTagModel.version());

        assertTrue(
                List.of(settings.getModels()).stream()
                        .noneMatch(model -> model.modelPath().toString().endsWith("apriltagV4-yolo11.rknn")));
    }

    @Test
    void getDefaultAprilTagModelDoesNotChangeObjectDetectionDefault() {
        var manager = NeuralNetworkModelManager.getInstance(true);
        manager.supportedBackends.clear();
        manager.supportedBackends.add(Family.RUBIK);

        var fuelModel =
                new StubModel(
                        new ModelProperties(
                                Path.of("fuelV1-yolo11n.tflite"),
                                "Fuel v11n",
                                List.of("Fuel"),
                                640,
                                640,
                                Family.RUBIK,
                                Version.YOLOV11));
        var aprilTagModel =
                new StubModel(
                        new ModelProperties(
                                Path.of("apriltagV4-yolo11.tflite"),
                                "AprilTag V4",
                                List.of("AprilTag"),
                                640,
                                640,
                                Family.RUBIK,
                                Version.YOLOV11));

        manager.models = new HashMap<>();
        manager.models.put(Family.RUBIK, new ArrayList<>(List.of(fuelModel, aprilTagModel)));

        assertSame(fuelModel, manager.getDefaultModel().orElseThrow());
        assertSame(aprilTagModel, manager.getDefaultAprilTagModel().orElseThrow());
    }

    private record StubModel(ModelProperties properties) implements Model {
        @Override
        public ObjectDetector load() {
            return NullModel.getInstance();
        }

        @Override
        public String getUID() {
            return properties.modelPath().toString();
        }

        @Override
        public String getNickname() {
            return properties.nickname();
        }

        @Override
        public Family getFamily() {
            return properties.family();
        }

        @Override
        public ModelProperties getProperties() {
            return properties;
        }
    }
}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
cmd /c '.\gradlew.bat --no-daemon -Dorg.gradle.java.home=C:\PROGRA~1\AMAZON~1\JDK170~1.17_ -Dorg.gradle.java.installations.paths=C:\PROGRA~1\AMAZON~1\JDK170~1.17_ -Dorg.gradle.java.installations.auto-detect=false :photon-core:test --tests org.photonvision.common.configuration.NeuralNetworkModelManagerTest'
```

Expected: compile failure because `getDefaultAprilTagModel()` does not exist, or assertion failure because `AprilTag V4` is not in shipped properties.

- [ ] **Step 3: Add model metadata and lookup**

In `NeuralNetworkModelManager.java`, add this `getShippedProperties` entry after the existing `fuelV1-yolo11n.tflite` Rubik model:

```java
        nnProps.addModelProperties(
                new ModelProperties(
                        Path.of(modelsDirectory.getAbsolutePath(), "apriltagV4-yolo11.tflite"),
                        "AprilTag V4",
                        new LinkedList<String>(List.of("AprilTag")),
                        640,
                        640,
                        Family.RUBIK,
                        Version.YOLOV11));
```

Add this method after `getDefaultModel()`:

```java
    /** The default AprilTag ROI model when ML-assisted AprilTag detection is enabled. */
    public Optional<Model> getDefaultAprilTagModel() {
        if (models == null || supportedBackends.isEmpty()) {
            return Optional.empty();
        }

        for (Family backend : supportedBackends) {
            if (!models.containsKey(backend)) {
                continue;
            }

            var model =
                    models.get(backend).stream()
                            .filter(m -> m.getNickname().toLowerCase().contains("apriltag"))
                            .findFirst();
            if (model.isPresent()) {
                return model;
            }
        }

        return Optional.empty();
    }
```

- [ ] **Step 4: Add the Rubik AprilTag TFLite model**

Run:

```powershell
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' restore --source=refs/remotes/doctorfogarty/apriltag-ml-experimental-sync --worktree -- photon-server/src/main/resources/models/apriltagV4-yolo11.tflite
```

Expected: `photon-server/src/main/resources/models/apriltagV4-yolo11.tflite` exists and no `.rknn` AprilTag model exists.

- [ ] **Step 5: Run model tests and commit**

Run:

```powershell
cmd /c '.\gradlew.bat --no-daemon -Dorg.gradle.java.home=C:\PROGRA~1\AMAZON~1\JDK170~1.17_ -Dorg.gradle.java.installations.paths=C:\PROGRA~1\AMAZON~1\JDK170~1.17_ -Dorg.gradle.java.installations.auto-detect=false :photon-core:test --tests org.photonvision.common.configuration.NeuralNetworkModelManagerTest'
```

Expected: `BUILD SUCCESSFUL`.

Commit:

```powershell
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' add photon-core/src/main/java/org/photonvision/common/configuration/NeuralNetworkModelManager.java photon-core/src/test/java/org/photonvision/common/configuration/NeuralNetworkModelManagerTest.java photon-server/src/main/resources/models/apriltagV4-yolo11.tflite
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' commit -m "Add Rubik AprilTag ML model"
```

---

### Task 2: ROI Detection And Decode Pipes

**Files:**
- Create: `photon-core/src/main/java/org/photonvision/vision/pipe/impl/MLDetectionResult.java`
- Create: `photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagROIDetectionPipe.java`
- Create: `photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagROIDecodePipe.java`
- Create: `photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagMLHybridPipe.java`
- Create: `photon-core/src/test/java/org/photonvision/vision/pipe/AprilTagROIDecodePipeTest.java`
- Create: `photon-core/src/test/java/org/photonvision/vision/pipe/AprilTagMLHybridPipeTest.java`

- [ ] **Step 1: Write failing host-only ROI tests**

Create `photon-core/src/test/java/org/photonvision/vision/pipe/AprilTagMLHybridPipeTest.java`:

```java
package org.photonvision.vision.pipe;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.pipe.impl.AprilTagMLHybridPipe;

public class AprilTagMLHybridPipeTest {
    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
        ConfigManager.getInstance().load();
    }

    @Test
    public void isAvailableFalseWithoutModel() {
        var pipe = new AprilTagMLHybridPipe();
        assertFalse(pipe.isAvailable());

        var result = pipe.run(new Frame());
        assertNotNull(result.output);
        assertTrue(result.output.detections().isEmpty());
        assertTrue(result.output.rois().isEmpty());

        pipe.release();
    }

    @Test
    public void releaseCanBeCalledTwice() {
        var pipe = new AprilTagMLHybridPipe();
        pipe.release();
        pipe.release();
    }
}
```

Create `photon-core/src/test/java/org/photonvision/vision/pipe/AprilTagROIDecodePipeTest.java` with these tests first:

```java
package org.photonvision.vision.pipe;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import edu.wpi.first.apriltag.AprilTagDetection;
import edu.wpi.first.apriltag.AprilTagDetector;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.util.TestUtils;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.frame.provider.FileFrameProvider;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipe.impl.AprilTagROIDecodePipe;
import org.photonvision.vision.pipe.impl.AprilTagROIDecodePipe.ROIDecodeInput;
import org.photonvision.vision.pipe.impl.AprilTagROIDecodePipe.ROIDecodeParams;

public class AprilTagROIDecodePipeTest {
    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
        ConfigManager.getInstance().load();
    }

    private static RotatedRect roiFromXYWH(double x, double y, double w, double h) {
        return new RotatedRect(new Point(x + w / 2.0, y + h / 2.0), new Size(w, h), 0);
    }

    @Test
    public void coordinateMappingMatchesFullFrameDetection() {
        var frameProvider =
                new FileFrameProvider(
                        TestUtils.getApriltagImagePath(TestUtils.ApriltagTestImages.kTag1_640_480, false),
                        TestUtils.WPI2020Image.FOV,
                        TestUtils.get2020LifeCamCoeffs(false));

        var frame = frameProvider.get();
        var fullFrameDetector = new AprilTagDetector();
        fullFrameDetector.addFamily(AprilTagFamily.kTag36h11.getNativeName());

        AprilTagDetection[] fullFrameDetections = fullFrameDetector.detect(frame.processedImage.getMat());
        assumeTrue(fullFrameDetections.length > 0, "full-frame detector found no tag");
        var groundTruth = fullFrameDetections[0];

        double minX = Double.MAX_VALUE;
        double maxX = Double.MIN_VALUE;
        double minY = Double.MAX_VALUE;
        double maxY = Double.MIN_VALUE;
        for (int i = 0; i < 4; i++) {
            minX = Math.min(minX, groundTruth.getCornerX(i));
            maxX = Math.max(maxX, groundTruth.getCornerX(i));
            minY = Math.min(minY, groundTruth.getCornerY(i));
            maxY = Math.max(maxY, groundTruth.getCornerY(i));
        }

        var pipe = new AprilTagROIDecodePipe();
        var params = new ROIDecodeParams();
        params.tagFamily = AprilTagFamily.kTag36h11;
        params.maxHammingDistance = 0;
        params.minDecisionMargin = 35;
        pipe.setParams(params);

        var roi = roiFromXYWH(minX - 20, minY - 20, maxX - minX + 40, maxY - minY + 40);
        var result = pipe.run(new ROIDecodeInput(frame.processedImage, List.of(roi)));

        assertEquals(1, result.output.size());
        var mapped = result.output.get(0);
        for (int i = 0; i < 4; i++) {
            assertEquals(groundTruth.getCornerX(i), mapped.getCornerX(i), 0.5);
            assertEquals(groundTruth.getCornerY(i), mapped.getCornerY(i), 0.5);
        }
        assertEquals(groundTruth.getCenterX(), mapped.getCenterX(), 0.5);
        assertEquals(groundTruth.getCenterY(), mapped.getCenterY(), 0.5);
        assertEquals(groundTruth.getId(), mapped.getId());

        fullFrameDetector.close();
        pipe.release();
    }

    @Test
    public void emptyRoiListReturnsEmptyResult() {
        var mat = Mat.zeros(480, 640, CvType.CV_8UC1);
        var pipe = new AprilTagROIDecodePipe();
        var params = new ROIDecodeParams();
        params.tagFamily = AprilTagFamily.kTag36h11;
        pipe.setParams(params);

        var result = pipe.run(new ROIDecodeInput(new CVMat(mat), new ArrayList<>()));

        assertNotNull(result.output);
        assertTrue(result.output.isEmpty());
        mat.release();
        pipe.release();
    }

    @Test
    public void expandBboxClampsToFrame() {
        var expanded =
                AprilTagROIDecodePipe.expandBbox(
                        roiFromXYWH(600, 440, 30, 30),
                        80,
                        640,
                        480);

        assertTrue(expanded.boundingRect().x >= 0);
        assertTrue(expanded.boundingRect().y >= 0);
        assertTrue(expanded.boundingRect().x + expanded.boundingRect().width <= 640);
        assertTrue(expanded.boundingRect().y + expanded.boundingRect().height <= 480);
    }
}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
cmd /c '.\gradlew.bat --no-daemon -Dorg.gradle.java.home=C:\PROGRA~1\AMAZON~1\JDK170~1.17_ -Dorg.gradle.java.installations.paths=C:\PROGRA~1\AMAZON~1\JDK170~1.17_ -Dorg.gradle.java.installations.auto-detect=false :photon-core:test --tests org.photonvision.vision.pipe.AprilTagMLHybridPipeTest --tests org.photonvision.vision.pipe.AprilTagROIDecodePipeTest'
```

Expected: compile failure because the new pipe classes do not exist.

- [ ] **Step 3: Port pipe source files from DoctorFogarty branch**

Use these files from `refs/remotes/doctorfogarty/apriltag-ml-experimental-sync` as source material:

```text
photon-core/src/main/java/org/photonvision/vision/pipe/impl/MLDetectionResult.java
photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagROIDetectionPipe.java
photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagROIDecodePipe.java
photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagMLHybridPipe.java
```

Apply these 2026 adaptations while creating the files:

```text
Replace org.wpilib.vision.apriltag.AprilTagDetection with edu.wpi.first.apriltag.AprilTagDetection.
Replace org.wpilib.vision.apriltag.AprilTagDetector with edu.wpi.first.apriltag.AprilTagDetector.
Keep java.util.List, org.opencv.core.RotatedRect, and existing PhotonVision package names unchanged.
Keep AprilTagMLHybridPipe.release() idempotent by guarding child pipe releases against null resources.
Keep AprilTagROIDetectionPipe.process() returning an empty list when detector is null, NullModel, params is null, or input mat is empty.
```

The `MLDetectionResult` content should be:

```java
package org.photonvision.vision.pipe.impl;

import edu.wpi.first.apriltag.AprilTagDetection;
import java.util.List;
import org.opencv.core.RotatedRect;

public record MLDetectionResult(List<AprilTagDetection> detections, List<RotatedRect> rois) {}
```

- [ ] **Step 4: Run ROI tests and commit**

Run:

```powershell
cmd /c '.\gradlew.bat --no-daemon -Dorg.gradle.java.home=C:\PROGRA~1\AMAZON~1\JDK170~1.17_ -Dorg.gradle.java.installations.paths=C:\PROGRA~1\AMAZON~1\JDK170~1.17_ -Dorg.gradle.java.installations.auto-detect=false :photon-core:test --tests org.photonvision.vision.pipe.AprilTagMLHybridPipeTest --tests org.photonvision.vision.pipe.AprilTagROIDecodePipeTest'
```

Expected: `BUILD SUCCESSFUL`.

Commit:

```powershell
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' add photon-core/src/main/java/org/photonvision/vision/pipe/impl/MLDetectionResult.java photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagROIDetectionPipe.java photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagROIDecodePipe.java photon-core/src/main/java/org/photonvision/vision/pipe/impl/AprilTagMLHybridPipe.java photon-core/src/test/java/org/photonvision/vision/pipe/AprilTagMLHybridPipeTest.java photon-core/src/test/java/org/photonvision/vision/pipe/AprilTagROIDecodePipeTest.java
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' commit -m "Add AprilTag ML ROI pipes"
```

---

### Task 3: AprilTag Pipeline Wiring And ROI Drawing

**Files:**
- Modify: `photon-core/src/main/java/org/photonvision/vision/pipeline/AprilTagPipelineSettings.java`
- Modify: `photon-core/src/main/java/org/photonvision/vision/pipeline/AprilTagPipeline.java`
- Modify: `photon-core/src/main/java/org/photonvision/vision/pipeline/result/CVPipelineResult.java`
- Create: `photon-core/src/main/java/org/photonvision/vision/pipe/impl/DrawMLROIPipe.java`
- Modify: `photon-core/src/main/java/org/photonvision/vision/pipeline/OutputStreamPipeline.java`
- Modify: `photon-core/src/main/java/org/photonvision/vision/processes/VisionModule.java`
- Create: `photon-core/src/test/java/org/photonvision/vision/pipeline/AprilTagPipelineMLFlagSafetyTest.java`

- [ ] **Step 1: Write failing pipeline safety tests**

Create `photon-core/src/test/java/org/photonvision/vision/pipeline/AprilTagPipelineMLFlagSafetyTest.java`:

```java
package org.photonvision.vision.pipeline;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Family;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Version;
import org.photonvision.common.configuration.NeuralNetworkModelsSettings.ModelProperties;
import org.photonvision.common.util.TestUtils;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.provider.FileFrameProvider;
import org.photonvision.vision.target.TargetModel;

public class AprilTagPipelineMLFlagSafetyTest {
    @BeforeEach
    public void setup() {
        LoadJNI.loadLibraries();
        ConfigManager.getInstance().load();
    }

    @Test
    public void mlSettingsParticipateInEquality() {
        var settings1 = new AprilTagPipelineSettings();
        var settings2 = new AprilTagPipelineSettings();

        assertEquals(settings1, settings2);

        settings1.useMLDetection = true;
        assertNotEquals(settings1, settings2);

        settings2.useMLDetection = true;
        assertEquals(settings1, settings2);

        settings1.mlConfidenceThreshold = 0.7;
        assertNotEquals(settings1, settings2);

        settings2.mlConfidenceThreshold = 0.7;
        assertEquals(settings1, settings2);

        settings1.model =
                new ModelProperties(
                        Path.of("test", "custom-apriltag.tflite").toAbsolutePath(),
                        "AprilTag custom",
                        List.of("AprilTag"),
                        640,
                        640,
                        Family.RUBIK,
                        Version.YOLOV11);
        assertNotEquals(settings1, settings2);

        settings2.model =
                new ModelProperties(
                        Path.of("test", "custom-apriltag.tflite").toAbsolutePath(),
                        "AprilTag custom",
                        List.of("AprilTag"),
                        640,
                        640,
                        Family.RUBIK,
                        Version.YOLOV11);
        assertEquals(settings1, settings2);
    }

    @Test
    public void mlEnabledOnNonRubikFallsBackToTraditionalDetector() {
        var pipeline = new AprilTagPipeline();

        pipeline.getSettings().useMLDetection = true;
        pipeline.getSettings().inputShouldShow = true;
        pipeline.getSettings().outputShouldDraw = true;
        pipeline.getSettings().solvePNPEnabled = true;
        pipeline.getSettings().targetModel = TargetModel.kAprilTag6p5in_36h11;
        pipeline.getSettings().tagFamily = AprilTagFamily.kTag36h11;

        var frameProvider =
                new FileFrameProvider(
                        TestUtils.getApriltagImagePath(TestUtils.ApriltagTestImages.kTag1_640_480, false),
                        TestUtils.WPI2020Image.FOV,
                        TestUtils.get2020LifeCamCoeffs(false));
        frameProvider.requestFrameThresholdType(pipeline.getThresholdType());

        var result = pipeline.run(frameProvider.get(), QuirkyCamera.DefaultCamera);

        assertFalse(result.targets.isEmpty());
        assertEquals(1, result.targets.size());
        assertEquals(1, result.targets.get(0).getFiducialId());
        assertNotNull(result.mlDetectionRois);
        assertEquals(0, result.mlDetectionRois.size());

        pipeline.release();
    }

    @Test
    public void traditionalDetectionStillWorksWhenMlDisabled() {
        var pipeline = new AprilTagPipeline();

        pipeline.getSettings().useMLDetection = false;
        pipeline.getSettings().inputShouldShow = true;
        pipeline.getSettings().outputShouldDraw = true;
        pipeline.getSettings().solvePNPEnabled = true;
        pipeline.getSettings().targetModel = TargetModel.kAprilTag6p5in_36h11;
        pipeline.getSettings().tagFamily = AprilTagFamily.kTag36h11;

        var frameProvider =
                new FileFrameProvider(
                        TestUtils.getApriltagImagePath(TestUtils.ApriltagTestImages.kTag1_640_480, false),
                        TestUtils.WPI2020Image.FOV,
                        TestUtils.get2020LifeCamCoeffs(false));
        frameProvider.requestFrameThresholdType(pipeline.getThresholdType());

        var result = pipeline.run(frameProvider.get(), QuirkyCamera.DefaultCamera);

        assertFalse(result.targets.isEmpty());
        assertEquals(1, result.targets.size());
        assertNotNull(result.targets.get(0).getBestCameraToTarget3d());

        pipeline.release();
    }
}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
cmd /c '.\gradlew.bat --no-daemon -Dorg.gradle.java.home=C:\PROGRA~1\AMAZON~1\JDK170~1.17_ -Dorg.gradle.java.installations.paths=C:\PROGRA~1\AMAZON~1\JDK170~1.17_ -Dorg.gradle.java.installations.auto-detect=false :photon-core:test --tests org.photonvision.vision.pipeline.AprilTagPipelineMLFlagSafetyTest'
```

Expected: compile failure for missing ML settings or `mlDetectionRois`.

- [ ] **Step 3: Add ML settings to `AprilTagPipelineSettings`**

Add imports:

```java
import java.util.Objects;
import org.photonvision.common.configuration.NeuralNetworkModelManager;
import org.photonvision.common.configuration.NeuralNetworkModelsSettings.ModelProperties;
import org.photonvision.vision.objects.Model;
```

Add fields after `doSingleTargetAlways`:

```java
    public boolean useMLDetection = false;
    public double mlConfidenceThreshold = 0.5;
    public double mlNmsThreshold = 0.45;
    public int mlRoiPaddingPixels = 40;
    public ModelProperties model =
            NeuralNetworkModelManager.getInstance()
                    .getDefaultAprilTagModel()
                    .map(Model::getProperties)
                    .orElse(null);
    public boolean showDetectionBoxes = true;
    public boolean atrEnabled = true;
    public int atrTargetDimension = 200;
    public double atrMinScaleFactor = 0.25;
```

Update `hashCode()` with all new fields. Use this model expression to avoid null crashes:

```java
        result = prime * result + ((model == null || model.modelPath() == null) ? 0 : model.modelPath().hashCode());
```

Update `equals()` with all new fields. Use:

```java
        if (!Objects.equals(model, other.model)) return false;
```

- [ ] **Step 4: Add ROI list to pipeline result**

In `CVPipelineResult.java`, add:

```java
import org.opencv.core.RotatedRect;
```

Add field:

```java
    public final List<RotatedRect> mlDetectionRois;
```

Update constructors so existing calls delegate to a full constructor with `List.of()` for ROI list. The full constructor must end with:

```java
            Frame inputFrame,
            List<String> classNames,
            List<RotatedRect> mlDetectionRois) {
        this.sequenceID = sequenceID;
        this.processingNanos = processingNanos;
        this.fps = fps;
        this.targets = targets != null ? targets : Collections.emptyList();
        this.multiTagResult = multiTagResult;
        this.objectDetectionClassNames = classNames;
        this.mlDetectionRois = mlDetectionRois != null ? mlDetectionRois : List.of();
        this.inputAndOutputFrame = inputFrame;
    }
```

- [ ] **Step 5: Wire `AprilTagPipeline`**

Add imports:

```java
import org.opencv.core.RotatedRect;
import org.photonvision.common.configuration.NeuralNetworkModelManager;
import org.photonvision.common.hardware.Platform;
import org.photonvision.vision.objects.Model;
import org.photonvision.vision.pipe.impl.AprilTagMLHybridPipe;
import org.photonvision.vision.pipe.impl.AprilTagROIDecodePipe;
import org.photonvision.vision.pipe.impl.AprilTagROIDetectionPipe;
```

Add fields:

```java
    private final AprilTagMLHybridPipe mlHybridPipe = new AprilTagMLHybridPipe();
    private boolean mlAvailable = false;
    private boolean mlWasAvailable = false;
```

In `setPipeParamsImpl()`, after normal AprilTag detector and pose pipe configuration, add a Rubik-only ML configuration block:

```java
        if (settings.useMLDetection) {
            boolean platformOk = Platform.getCurrentPlatform() == Platform.LINUX_QCS6490;
            Model aprilTagModel = null;
            if (platformOk) {
                if (settings.model != null && settings.model.modelPath() != null) {
                    aprilTagModel =
                            NeuralNetworkModelManager.getInstance()
                                    .getModel(settings.model.modelPath().toString())
                                    .orElse(null);
                }
                if (aprilTagModel == null) {
                    aprilTagModel =
                            NeuralNetworkModelManager.getInstance().getDefaultAprilTagModel().orElse(null);
                }
            }

            if (platformOk && aprilTagModel != null) {
                var detectionParams =
                        new AprilTagROIDetectionPipe.AprilTagROIDetectionParams(
                                aprilTagModel, settings.mlConfidenceThreshold, settings.mlNmsThreshold);

                var decodeParams = new AprilTagROIDecodePipe.ROIDecodeParams();
                decodeParams.tagFamily = settings.tagFamily;
                decodeParams.maxHammingDistance = settings.hammingDist;
                decodeParams.minDecisionMargin = settings.decisionMargin;
                decodeParams.detectorConfig.numThreads = settings.threads;
                decodeParams.detectorConfig.refineEdges = settings.refineEdges;
                decodeParams.detectorConfig.quadDecimate = 1;
                decodeParams.detectorConfig.quadSigma = (float) settings.blur;
                decodeParams.atrEnabled = settings.atrEnabled;
                decodeParams.atrTargetDimension = settings.atrTargetDimension;
                decodeParams.atrMinScaleFactor = settings.atrMinScaleFactor;

                mlHybridPipe.setParams(
                        new AprilTagMLHybridPipe.Params(
                                detectionParams, decodeParams, settings.mlRoiPaddingPixels));
            }

            mlAvailable = platformOk && aprilTagModel != null && mlHybridPipe.isAvailable();
            if (!mlWasAvailable && mlAvailable) {
                logger.info("Rubik ML-assisted AprilTag detection enabled");
            }
            if (platformOk && aprilTagModel == null && mlWasAvailable) {
                logger.warn("ML-assisted AprilTag detection enabled but no AprilTag model was found");
            }
        } else {
            mlAvailable = false;
        }
        mlWasAvailable = mlAvailable;
```

In `process()`, replace the single traditional detection call with:

```java
        List<AprilTagDetection> detections;
        long detectionNanos;
        List<RotatedRect> mlDetectionRois = List.of();

        if (settings.useMLDetection && mlAvailable) {
            var hybridResult = mlHybridPipe.run(frame);
            detections = hybridResult.output.detections();
            detectionNanos = hybridResult.nanosElapsed;
            mlDetectionRois = hybridResult.output.rois();
        } else {
            CVPipeResult<List<AprilTagDetection>> tagDetectionPipeResult =
                    aprilTagDetectionPipe.run(frame.processedImage);
            detections = tagDetectionPipeResult.output;
            detectionNanos = tagDetectionPipeResult.nanosElapsed;
        }
        sumPipeNanosElapsed += detectionNanos;
```

Return the result with ROI list:

```java
        return new CVPipelineResult(
                frame.sequenceID,
                sumPipeNanosElapsed,
                fps,
                targetList,
                multiTagResult,
                frame,
                List.of(),
                mlDetectionRois);
```

Update `release()`:

```java
        mlHybridPipe.release();
```

- [ ] **Step 6: Add ROI drawing pipe and stream propagation**

Create `DrawMLROIPipe.java` from DoctorFogarty's file with these 2026 adaptations:

```text
Use import edu.wpi.first.math.Pair.
Use org.opencv.core.MatOfPoint, Point, RotatedRect, Scalar.
Draw cyan ROI polylines only when shouldDraw and showDetectionBoxes are true.
```

In `OutputStreamPipeline.java`:

```java
import org.opencv.core.RotatedRect;
```

Add field:

```java
    private final DrawMLROIPipe drawMLROIPipe = new DrawMLROIPipe();
```

Add AprilTag params in `setPipeParams()`:

```java
        if (settings instanceof AprilTagPipelineSettings atSettings) {
            drawMLROIPipe.setParams(
                    new DrawMLROIPipe.DrawMLROIParams(
                            settings.outputShouldDraw,
                            atSettings.showDetectionBoxes,
                            settings.streamingFrameDivisor));
        }
```

Keep existing public method and delegate to an overload:

```java
    public CVPipelineResult process(
            Frame inputAndOutputFrame,
            AdvancedPipelineSettings settings,
            List<TrackedTarget> targetsToDraw) {
        return process(inputAndOutputFrame, settings, targetsToDraw, List.of());
    }
```

Add overload with ROI list:

```java
    public CVPipelineResult process(
            Frame inputAndOutputFrame,
            AdvancedPipelineSettings settings,
            List<TrackedTarget> targetsToDraw,
            List<RotatedRect> mlDetectionRois) {
```

Inside the AprilTag branch before drawing AprilTag markers:

```java
                drawMLROIPipe.run(Pair.of(outMat, mlDetectionRois));
```

In `VisionModule.StreamRunnable`, add `List<RotatedRect> mlDetectionRois = List.of();`, include it in `updateData(...)`, copy it under lock, and pass it into:

```java
CVPipelineResult osr = outputStreamPipeline.process(m_frame, settings, targets, mlDetectionRois);
```

In `VisionModule.consumeResult(...)`, update:

```java
streamRunnable.updateData(result.inputAndOutputFrame, settings, result.targets, result.mlDetectionRois);
```

- [ ] **Step 7: Run pipeline tests and commit**

Run:

```powershell
cmd /c '.\gradlew.bat --no-daemon -Dorg.gradle.java.home=C:\PROGRA~1\AMAZON~1\JDK170~1.17_ -Dorg.gradle.java.installations.paths=C:\PROGRA~1\AMAZON~1\JDK170~1.17_ -Dorg.gradle.java.installations.auto-detect=false :photon-core:test --tests org.photonvision.vision.pipeline.AprilTagPipelineMLFlagSafetyTest --tests org.photonvision.vision.pipeline.AprilTagTest'
```

Expected: `BUILD SUCCESSFUL`.

Commit:

```powershell
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' add photon-core/src/main/java/org/photonvision/vision/pipeline/AprilTagPipelineSettings.java photon-core/src/main/java/org/photonvision/vision/pipeline/AprilTagPipeline.java photon-core/src/main/java/org/photonvision/vision/pipeline/result/CVPipelineResult.java photon-core/src/main/java/org/photonvision/vision/pipe/impl/DrawMLROIPipe.java photon-core/src/main/java/org/photonvision/vision/pipeline/OutputStreamPipeline.java photon-core/src/main/java/org/photonvision/vision/processes/VisionModule.java photon-core/src/test/java/org/photonvision/vision/pipeline/AprilTagPipelineMLFlagSafetyTest.java
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' commit -m "Wire Rubik ML AprilTag detection"
```

---

### Task 4: AprilTag UI Controls

**Files:**
- Modify: `photon-client/src/types/PipelineTypes.ts`
- Modify: `photon-client/src/components/dashboard/tabs/AprilTagTab.vue`

- [ ] **Step 1: Add frontend settings types**

In `photon-client/src/types/PipelineTypes.ts`, extend `AprilTagPipelineSettings`:

```ts
  useMLDetection: boolean;
  mlConfidenceThreshold: number;
  mlNmsThreshold: number;
  mlRoiPaddingPixels: number;
  model: ObjectDetectionModelProperties | null;
  showDetectionBoxes: boolean;
  atrEnabled: boolean;
  atrTargetDimension: number;
  atrMinScaleFactor: number;
```

Extend `DefaultAprilTagPipelineSettings`:

```ts
  useMLDetection: false,
  mlConfidenceThreshold: 0.5,
  mlNmsThreshold: 0.45,
  mlRoiPaddingPixels: 40,
  model: null,
  showDetectionBoxes: true,
  atrEnabled: true,
  atrTargetDimension: 200,
  atrMinScaleFactor: 0.25
```

- [ ] **Step 2: Add Rubik-only controls to `AprilTagTab.vue`**

Use existing `ObjectDetectionTab.vue` model-selection pattern. Add imports:

```ts
import { PipelineType, type AprilTagPipelineSettings } from "@/types/PipelineTypes";
import { useSettingsStore } from "@/stores/settings/GeneralSettingsStore";
import type { ObjectDetectionModelProperties } from "@/types/SettingTypes";
```

Change current settings computed:

```ts
const currentPipelineSettings = computed<AprilTagPipelineSettings>(
  () => useCameraSettingsStore().currentPipelineSettings as AprilTagPipelineSettings
);
```

Add computed model state:

```ts
const supportedAprilTagModels = computed<ObjectDetectionModelProperties[]>(() => {
  const { availableModels, supportedBackends } = useSettingsStore().general;
  const rubikSupported = supportedBackends.some((backend: string) => backend.toLowerCase() === "rubik");
  if (!rubikSupported) return [];

  return availableModels.filter(
    (model: ObjectDetectionModelProperties) =>
      model.family.toLowerCase() === "rubik" && model.nickname.toLowerCase().includes("apriltag")
  );
});

const mlDetectionAvailable = computed(() => supportedAprilTagModels.value.length > 0);

const selectedAprilTagModel = computed({
  get: () => {
    const currentModel = currentPipelineSettings.value.model;
    if (!currentModel) return undefined;

    const index = supportedAprilTagModels.value.findIndex((model) => model.modelPath === currentModel.modelPath);
    return index === -1 ? undefined : index;
  },
  set: (value) => {
    if (value !== undefined && value >= 0 && value < supportedAprilTagModels.value.length) {
      useCameraSettingsStore().changeCurrentPipelineSetting({ model: supportedAprilTagModels.value[value] }, true);
    }
  }
});
```

Add template section after `Refine Edges`:

```vue
    <v-divider v-if="mlDetectionAvailable" class="mt-3 mb-2" />
    <div v-if="mlDetectionAvailable">
      <p class="text-subtitle-2 mb-2">ML-Tag</p>
      <pv-switch
        v-model="currentPipelineSettings.useMLDetection"
        :switch-cols="interactiveCols"
        label="Enable ML-Tag"
        tooltip="Uses the Rubik Pi 3 NPU to find AprilTag regions before decoding tags."
        @update:modelValue="
          (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ useMLDetection: value }, false)
        "
      />
      <div v-if="currentPipelineSettings.useMLDetection">
        <pv-select
          v-model="selectedAprilTagModel"
          label="Model"
          tooltip="The Rubik model used to find AprilTag regions."
          :select-cols="interactiveCols"
          :items="supportedAprilTagModels.map((model) => model.nickname)"
        />
        <pv-slider
          v-model="currentPipelineSettings.mlConfidenceThreshold"
          :slider-cols="interactiveCols"
          label="Confidence"
          tooltip="Minimum ML confidence for ROI detection."
          :min="0"
          :max="1"
          :step="0.01"
          @update:modelValue="
            (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ mlConfidenceThreshold: value }, false)
          "
        />
        <pv-slider
          v-model="currentPipelineSettings.mlNmsThreshold"
          :slider-cols="interactiveCols"
          label="NMS Threshold"
          tooltip="Overlap threshold used to merge ML ROI detections."
          :min="0"
          :max="1"
          :step="0.01"
          @update:modelValue="
            (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ mlNmsThreshold: value }, false)
          "
        />
        <pv-slider
          v-model="currentPipelineSettings.mlRoiPaddingPixels"
          :slider-cols="interactiveCols"
          label="ROI Padding"
          tooltip="Pixels added around each detected AprilTag region."
          :min="0"
          :max="150"
          :step="5"
          @update:modelValue="
            (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ mlRoiPaddingPixels: value }, false)
          "
        />
        <pv-switch
          v-model="currentPipelineSettings.showDetectionBoxes"
          :switch-cols="interactiveCols"
          label="Show ROI Boxes"
          tooltip="Draws the ML ROI boxes on the processed stream."
          @update:modelValue="
            (value) => useCameraSettingsStore().changeCurrentPipelineSetting({ showDetectionBoxes: value }, false)
          "
        />
      </div>
    </div>
```

- [ ] **Step 3: Run frontend type/lint check available in repo**

Run:

```powershell
cmd /c '.\gradlew.bat --no-daemon -Dorg.gradle.java.home=C:\PROGRA~1\AMAZON~1\JDK170~1.17_ -Dorg.gradle.java.installations.paths=C:\PROGRA~1\AMAZON~1\JDK170~1.17_ -Dorg.gradle.java.installations.auto-detect=false :photon-client:installPnpm :photon-client:lint'
```

Expected: frontend lint passes. If this task name does not exist on this branch, run `cmd /c '.\gradlew.bat tasks --all'` and use the listed Photon client lint/typecheck task.

- [ ] **Step 4: Commit UI changes**

Commit:

```powershell
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' add photon-client/src/types/PipelineTypes.ts photon-client/src/components/dashboard/tabs/AprilTagTab.vue
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' commit -m "Add Rubik ML AprilTag UI controls"
```

---

### Task 5: Verification And Rubik Build

**Files:**
- Verify only. No planned source changes unless tests expose a concrete bug.

- [ ] **Step 1: Run focused Java tests**

Run:

```powershell
cmd /c '.\gradlew.bat --no-daemon -Dorg.gradle.java.home=C:\PROGRA~1\AMAZON~1\JDK170~1.17_ -Dorg.gradle.java.installations.paths=C:\PROGRA~1\AMAZON~1\JDK170~1.17_ -Dorg.gradle.java.installations.auto-detect=false :photon-core:test --tests org.photonvision.common.configuration.NeuralNetworkModelManagerTest --tests org.photonvision.vision.pipe.AprilTagMLHybridPipeTest --tests org.photonvision.vision.pipe.AprilTagROIDecodePipeTest --tests org.photonvision.vision.pipeline.AprilTagPipelineMLFlagSafetyTest --tests org.photonvision.vision.pipeline.AprilTagTest --tests org.photonvision.common.configuration.NetworkConfigTest --tests org.photonvision.common.networking.NetworkManagerTest --tests org.photonvision.vision.processes.VisionModuleCameraSettingsTest'
```

Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 2: Run whitespace/status checks**

Run:

```powershell
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' diff --check HEAD~4..HEAD
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' status -sb
```

Expected: no whitespace errors. Status should show only known unrelated `photon-client/pnpm-workspace.yaml` if Task 4 did not need it.

- [ ] **Step 3: Build linuxarm64 Rubik JAR**

Run the existing Rubik/linuxarm64 build command used for this branch. If unsure, inspect `.github/workflows/build.yml` and `scripts/armrunner.sh`, then run the same Gradle target used to produce `photon-server/build/libs/photonvision-v2026.3.4-linuxarm64*.jar`.

Expected:

```text
photon-server/build/libs/photonvision-v2026.3.4-linuxarm64*.jar exists
jar tf photon-server/build/libs/<jar> contains models/apriltagV4-yolo11.tflite
jar tf photon-server/build/libs/<jar> contains linux/arm64/shared/libphotontargetingJNI.so
```

- [ ] **Step 4: Rebuild Rubik Pi 3 dev image if requested**

Use the existing Rubik image procedure from the current branch. Verify the rebuilt image still contains:

```text
/opt/photonvision/photonvision.jar
/etc/NetworkManager/system-connections/static-team2375.nmconnection
/etc/NetworkManager/system-connections/debug-wlan0.nmconnection
```

Verify static profile contains:

```text
address1=10.23.75.14/8,10.23.75.4
autoconnect-priority=100
```

Verify Wi-Fi fallback profile contains:

```text
ssid=NinJAsPeeD
route-metric=600
autoconnect-priority=-100
```

- [ ] **Step 5: Push after user approval**

After verification, show the commit list and ask before pushing if the user did not already request push for this phase.

```powershell
git -c safe.directory='C:/FRC_Software/FRC 2026 Software/Photonvision Firmware/photonvision' log --oneline --decorate origin/2026-team2375..HEAD
```

Expected commits:

```text
Add Rubik AprilTag ML model
Add AprilTag ML ROI pipes
Wire Rubik ML AprilTag detection
Add Rubik ML AprilTag UI controls
```

## Self-Review

Spec coverage:

- Rubik-only model: Task 1.
- No `.rknn`: Task 1 test verifies no AprilTag `.rknn` shipped property.
- Model lookup separate from object-detection default: Task 1 test verifies both defaults.
- ROI pipes and result container: Task 2.
- ML pipeline fallback and Rubik-only gate: Task 3.
- ROI visualization: Task 3.
- UI controls: Task 4.
- Focused tests, JAR build, image preservation: Task 5.

Type consistency:

- Java imports stay on `edu.wpi.first.*`.
- Frontend model type reuses `ObjectDetectionModelProperties`.
- ROI rectangles use `org.opencv.core.RotatedRect`.
- Output stream keeps the existing three-argument `process` method and adds a four-argument overload.

Scope check:

- No 2027 Java, Gradle, PhotonLib, examples, website, RKNN, or Orange Pi work is included.
