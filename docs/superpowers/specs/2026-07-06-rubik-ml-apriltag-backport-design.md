# Rubik Pi 3 ML AprilTag Backport Design

## Goal

Backport only DoctorFogarty's ML-assisted AprilTag ROI feature onto `2026-team2375` for Rubik Pi 3. Keep Java 17, PhotonVision `v2026.3.4`, Team 2375 static IP defaults, and the Rubik Pi 3 image workflow intact.

## Source Branch

Use `DoctorFogarty/photonvision` branch `apriltag-ml-experimental-sync` as reference only. Do not merge it directly because it includes broad 2027/main work: Java 25, Gradle changes, WPILib package namespace changes, PhotonLib API rewrites, examples, docs, and UI churn.

## Scope

Include the Rubik Pi 3 ML AprilTag feature slice:

- Package `apriltagV4-yolo11.tflite` as the only new AprilTag ML model.
- Add a shipped `Family.RUBIK` / `YOLOV11` AprilTag model entry named `AprilTag V4`.
- Add `getDefaultAprilTagModel()` or equivalent lookup that selects AprilTag models without affecting the general object-detection default.
- Add AprilTag ML pipeline settings for enablement, confidence threshold, NMS threshold, ROI padding, ROI box drawing, and adaptive tag resizing.
- Add the ML ROI detect/decode pipes and result container.
- Modify `AprilTagPipeline` to run the ML-assisted path only when enabled, running on `Platform.LINUX_QCS6490`, and backed by a valid Rubik model.
- Preserve the traditional AprilTag detection path as the default and as the fallback when ML is disabled, unsupported, or unavailable.
- Add ROI box visualization to the output stream path.
- Add AprilTag dashboard controls using the existing 2026 frontend patterns without importing unrelated frontend type churn.
- Add focused tests for settings, fallback behavior, ROI expansion, homography/corner mapping, and traditional pipeline preservation.

Exclude everything else:

- No `.rknn` model.
- No Orange Pi / RKNN support.
- No Java 25 or Gradle upgrade.
- No WPILib package namespace migration.
- No PhotonLib API rewrite.
- No examples, docs, website, or unrelated CI changes.
- No direct merge from DoctorFogarty branch.

## Architecture

The AprilTag pipeline gets a second detector path in front of the existing pose-estimation logic. The ML path uses the Rubik object detector to find candidate AprilTag ROIs on the color frame, pads each ROI, then runs WPILib's normal AprilTag decoder on the grayscale ROI. Detections are mapped back into full-frame coordinates before the existing single-tag and multi-tag pose code consumes them.

The pipeline chooses between detectors at runtime:

- Default: traditional full-frame AprilTag detector.
- ML enabled and available: Rubik model ROI detector plus ROI decoder.
- ML enabled but unavailable: traditional detector fallback.

This keeps pose estimation, target publication, and Team 2375 network/image behavior stable.

## Backend Gating

ML AprilTag acceleration is Rubik-only for this backport. Runtime gating must require `Platform.LINUX_QCS6490`. UI controls should appear only when the backend list contains `RUBIK` and a supported AprilTag model exists.

The model manager should continue supporting existing COCO/Fuel object detection models. The AprilTag model should not become the default object detector model for the Object Detection pipeline unless the user explicitly selects it there.

## Data Flow

1. Camera frame enters `AprilTagPipeline` with grayscale processed image and color source image.
2. `AprilTagPipeline` builds AprilTag detector config from existing settings.
3. If ML is disabled or unavailable, run `AprilTagDetectionPipe` unchanged.
4. If ML is enabled and available:
   - `AprilTagROIDetectionPipe` runs Rubik YOLO model on the color frame.
   - `AprilTagMLHybridPipe` pads returned ROIs and passes them to `AprilTagROIDecodePipe`.
   - `AprilTagROIDecodePipe` decodes tags in each ROI, maps corners and homography into full-frame coordinates, filters by hamming and decision margin, and deduplicates by tag ID.
5. Existing multi-tag and single-tag pose estimation receives the resulting full-frame detections.
6. `CVPipelineResult` carries optional ROI rectangles for visualization only.
7. `OutputStreamPipeline` draws ROI boxes when enabled.

## Error Handling

The feature must fail closed:

- Missing model file logs a warning and uses traditional detection.
- Unsupported platform uses traditional detection.
- Empty ROI list returns no ML detections and does not crash.
- Invalid or out-of-bounds ROI clamps to frame bounds.
- Null model/settings entries do not throw during pipeline hash/equality checks.
- Pipeline release closes both traditional and ML resources.

## Testing

Focused verification should include:

- `AprilTagPipelineSettings` equality/hash tests for new ML fields.
- Non-Rubik fallback test: enabling ML on the Windows dev host still detects the standard test AprilTag through the traditional path.
- Traditional disabled-ML test: current AprilTag detection behavior remains intact.
- ROI expansion and clamp unit tests.
- ROI coordinate mapping tests for corners and homography, including adaptive tag resizing scale.
- Model manager test that Rubik AprilTag model properties exist when shipped models are discovered.
- Focused Gradle run for the new tests plus existing AprilTag tests.
- Rubik linuxarm64 server JAR build.
- Rubik image rebuild check that the Team 2375 static IP and WLAN debug profiles remain present.

## Commit Plan

Use small commits:

1. Add Rubik AprilTag model metadata and packaged `.tflite`.
2. Add ROI detection/decode pipes and unit tests.
3. Wire ML fallback into AprilTag pipeline and result/output drawing.
4. Add AprilTag UI controls.
5. Run verification and rebuild Rubik image if requested.
