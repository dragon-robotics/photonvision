/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.vision.pipeline;

import edu.wpi.first.apriltag.AprilTagDetection;
import edu.wpi.first.apriltag.AprilTagDetector;
import edu.wpi.first.apriltag.AprilTagPoseEstimate;
import edu.wpi.first.apriltag.AprilTagPoseEstimator.Config;
import edu.wpi.first.math.geometry.CoordinateSystem;
import edu.wpi.first.math.geometry.Pose3d;
import edu.wpi.first.math.geometry.Rotation3d;
import edu.wpi.first.math.geometry.Transform3d;
import edu.wpi.first.math.util.Units;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.opencv.core.Mat;
import org.opencv.core.RotatedRect;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.configuration.NeuralNetworkModelManager;
import org.photonvision.common.dataflow.structures.Packet;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.math.MathUtils;
import org.photonvision.estimation.TargetModel;
import org.photonvision.targeting.MultiTargetPNPResult;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.calibration.CameraCalibrationCoefficients;
import org.photonvision.vision.calibration.JsonMatOfDouble;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.objects.Model;
import org.photonvision.vision.pipe.impl.AprilTagDetectionCudaPipe;
import org.photonvision.vision.pipe.impl.AprilTagDetectionCudaPipe.AprilTagDetectionCudaPipeParams;
import org.photonvision.vision.pipe.impl.AprilTagDetectionCudaPipe.Calibration;
import org.photonvision.vision.pipe.impl.AprilTagDetectionPipe;
import org.photonvision.vision.pipe.impl.AprilTagDetectionPipe.AprilTagDetectionPipeParams;
import org.photonvision.vision.pipe.impl.AprilTagMLHybridPipe;
import org.photonvision.vision.pipe.impl.AprilTagPoseEstimatorPipe;
import org.photonvision.vision.pipe.impl.AprilTagPoseEstimatorPipe.AprilTagPoseEstimatorPipeParams;
import org.photonvision.vision.pipe.impl.AprilTagROIDecodePipe;
import org.photonvision.vision.pipe.impl.AprilTagROIDetectionPipe;
import org.photonvision.vision.pipe.impl.CalculateFPSPipe;
import org.photonvision.vision.pipe.impl.MultiTargetPNPPipe;
import org.photonvision.vision.pipe.impl.MultiTargetPNPPipe.MultiTargetPNPPipeParams;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.photonvision.vision.target.TrackedTarget;
import org.photonvision.vision.target.TrackedTarget.TargetCalculationParameters;

public class AprilTagPipeline extends CVPipeline<CVPipelineResult, AprilTagPipelineSettings> {
    private static final Logger logger = new Logger(AprilTagPipeline.class, LogGroup.VisionModule);
    private static final Calibration DEFAULT_CUDA_CALIBRATION =
            new Calibration(1, 1, 0, 0, 0, 0, 0, 0, 0);

    private final AprilTagDetectionPipe aprilTagDetectionPipe;
    private final AprilTagDetectionCudaPipe cudaDetectionPipe;
    private final AprilTagPoseEstimatorPipe singleTagPoseEstimatorPipe;
    private final MultiTargetPNPPipe multiTagPNPPipe;
    private final CalculateFPSPipe calculateFPSPipe;
    private final AprilTagMLHybridPipe mlHybridPipe;
    private FrameStaticProperties currentTargetCalculationProperties;
    private boolean currentPoseProcessingAllowed = true;

    private static final FrameThresholdType PROCESSING_TYPE = FrameThresholdType.GREYSCALE;

    public AprilTagPipeline() {
        this(new AprilTagPipelineSettings());
    }

    public AprilTagPipeline(AprilTagPipelineSettings settings) {
        this(
                settings,
                new AprilTagDetectionPipe(),
                new AprilTagDetectionCudaPipe(),
                new AprilTagMLHybridPipe(),
                new AprilTagPoseEstimatorPipe(),
                new MultiTargetPNPPipe(),
                new CalculateFPSPipe());
    }

    AprilTagPipeline(
            AprilTagPipelineSettings settings,
            AprilTagDetectionPipe aprilTagDetectionPipe,
            AprilTagDetectionCudaPipe cudaDetectionPipe,
            AprilTagMLHybridPipe mlHybridPipe,
            AprilTagPoseEstimatorPipe singleTagPoseEstimatorPipe,
            MultiTargetPNPPipe multiTagPNPPipe,
            CalculateFPSPipe calculateFPSPipe) {
        super(PROCESSING_TYPE);
        this.settings = settings;
        this.aprilTagDetectionPipe = aprilTagDetectionPipe;
        this.cudaDetectionPipe = cudaDetectionPipe;
        this.mlHybridPipe = mlHybridPipe;
        this.singleTagPoseEstimatorPipe = singleTagPoseEstimatorPipe;
        this.multiTagPNPPipe = multiTagPNPPipe;
        this.calculateFPSPipe = calculateFPSPipe;
    }

    Optional<Model> getMlModel() {
        if (settings.mlModelName != null && !settings.mlModelName.isBlank()) {
            return NeuralNetworkModelManager.getInstance().getModel(settings.mlModelName);
        }

        return NeuralNetworkModelManager.getInstance()
                .getDefaultModel()
                .filter(
                        model -> {
                            var properties = model.getProperties();
                            if (properties == null) {
                                return false;
                            }
                            return properties.labels().contains("AprilTag")
                                    || properties.nickname().toLowerCase().contains("apriltag");
                        });
    }

    @Override
    protected void setPipeParamsImpl() {
        boolean cudaEnabled =
                settings.useCudaTagDetection
                        && settings.tagFamily == AprilTagFamily.kTag36h11
                        && !hasUnsupportedCudaDistortion(frameStaticProperties.cameraCalibration);
        cudaDetectionPipe.setEnabled(cudaEnabled);
        if (cudaEnabled) {
            cudaDetectionPipe.setParams(new AprilTagDetectionCudaPipeParams(settings.decimate));
        }

        // Sanitize thread count - not supported to have fewer than 1 threads
        settings.threads = Math.max(1, settings.threads);

        // for now, hard code tag width based on enum value
        // From 2024 best guess is 6.5
        double tagWidth = Units.inchesToMeters(6.5);
        TargetModel tagModel = TargetModel.kAprilTag36h11;
        if (settings.tagFamily == AprilTagFamily.kTag16h5) {
            // 2023 tag, 6in
            tagWidth = Units.inchesToMeters(6);
            tagModel = TargetModel.kAprilTag16h5;
        }

        var config = new AprilTagDetector.Config();
        config.numThreads = settings.threads;
        config.refineEdges = settings.refineEdges;
        config.quadSigma = (float) settings.blur;
        config.quadDecimate = settings.decimate;

        var quadParams = new AprilTagDetector.QuadThresholdParameters();
        // 5 was the default minClusterPixels in WPILib prior to 2025
        // increasing it causes detection problems when decimate > 1
        quadParams.minClusterPixels = 5;
        // these are the same as the values in WPILib 2025
        // setting them here to prevent upstream changes from changing behavior of the detector
        quadParams.maxNumMaxima = 10;
        quadParams.criticalAngle = 45 * Math.PI / 180.0;
        quadParams.maxLineFitMSE = 10.0f;
        quadParams.minWhiteBlackDiff = 5;
        quadParams.deglitch = false;

        aprilTagDetectionPipe.setParams(
                new AprilTagDetectionPipeParams(settings.tagFamily, config, quadParams));

        if (settings.useMLDetection) {
            var selectedMlModel = getMlModel();
            var detectionParams =
                    new AprilTagROIDetectionPipe.AprilTagROIDetectionParams(
                            selectedMlModel.orElse(null),
                            settings.mlConfidenceThreshold,
                            settings.mlNmsThreshold);
            var decodeParams = new AprilTagROIDecodePipe.ROIDecodeParams();
            decodeParams.tagFamily = settings.tagFamily;
            decodeParams.detectorConfig = config;
            decodeParams.quadParams = quadParams;
            decodeParams.maxHammingDistance = settings.hammingDist;
            decodeParams.minDecisionMargin = settings.decisionMargin;

            mlHybridPipe.setParams(
                    new AprilTagMLHybridPipe.Params(
                            detectionParams, decodeParams, settings.mlRoiPaddingPixels));
        }

        if (cudaEnabled) {
            var validatedCalibration = getValidatedCalibration(frameStaticProperties.cameraCalibration);
            cudaDetectionPipe.setCalibration(validatedCalibration.orElse(DEFAULT_CUDA_CALIBRATION));
            currentPoseProcessingAllowed = validatedCalibration.isPresent();
            currentTargetCalculationProperties =
                    currentPoseProcessingAllowed
                            ? frameStaticProperties
                            : new FrameStaticProperties(
                                    frameStaticProperties.imageWidth,
                                    frameStaticProperties.imageHeight,
                                    frameStaticProperties.fov,
                                    null);
            if (validatedCalibration.isPresent()) {
                var calibration = validatedCalibration.get();
                setPoseEstimatorParams(
                        tagWidth,
                        tagModel,
                        frameStaticProperties.cameraCalibration,
                        calibration.fx(),
                        calibration.fy(),
                        calibration.cx(),
                        calibration.cy());
            }
        } else {
            currentPoseProcessingAllowed = true;
            currentTargetCalculationProperties = frameStaticProperties;
            if (frameStaticProperties.cameraCalibration != null) {
                var cameraMatrix = frameStaticProperties.cameraCalibration.getCameraIntrinsicsMat();
                if (cameraMatrix != null && cameraMatrix.rows() > 0) {
                    var cx = cameraMatrix.get(0, 2)[0];
                    var cy = cameraMatrix.get(1, 2)[0];
                    var fx = cameraMatrix.get(0, 0)[0];
                    var fy = cameraMatrix.get(1, 1)[0];
                    setPoseEstimatorParams(
                            tagWidth, tagModel, frameStaticProperties.cameraCalibration, fx, fy, cx, cy);
                }
            }
        }
    }

    private void setPoseEstimatorParams(
            double tagWidth,
            TargetModel tagModel,
            CameraCalibrationCoefficients calibration,
            double fx,
            double fy,
            double cx,
            double cy) {
        singleTagPoseEstimatorPipe.setParams(
                new AprilTagPoseEstimatorPipeParams(
                        new Config(tagWidth, fx, fy, cx, cy), calibration, settings.numIterations));

        // TODO global state ew
        var atfl = ConfigManager.getInstance().getConfig().getApriltagFieldLayout();
        multiTagPNPPipe.setParams(new MultiTargetPNPPipeParams(calibration, atfl, tagModel));
    }

    private static Optional<Calibration> getValidatedCalibration(
            CameraCalibrationCoefficients calibration) {
        if (calibration == null
                || !hasExpectedData(calibration.cameraIntrinsics)
                || !hasExpectedData(calibration.distCoeffs)) {
            return Optional.empty();
        }

        try {
            Mat intrinsics = calibration.getCameraIntrinsicsMat();
            Mat distortion = calibration.getDistCoeffsMat();
            int distortionCount = (int) distortion.total();
            if (intrinsics.rows() != 3
                    || intrinsics.cols() != 3
                    || intrinsics.channels() != 1
                    || (distortion.rows() != 1 && distortion.cols() != 1)
                    || distortion.channels() != 1
                    || !isSupportedDistortionCount(distortionCount)) {
                return Optional.empty();
            }

            double[] intrinsicData = new double[9];
            double[] distortionData = new double[distortionCount];
            if (intrinsics.get(0, 0, intrinsicData) != intrinsicData.length * Double.BYTES
                    || distortion.get(0, 0, distortionData) != distortionData.length * Double.BYTES
                    || !allFinite(intrinsicData)
                    || !allFinite(distortionData)
                    || intrinsicData[0] <= 0
                    || intrinsicData[4] <= 0) {
                return Optional.empty();
            }

            return Optional.of(
                    new Calibration(
                            intrinsicData[0],
                            intrinsicData[4],
                            intrinsicData[2],
                            intrinsicData[5],
                            distortionData[0],
                            distortionData[1],
                            distortionData[2],
                            distortionData[3],
                            distortionCount > 4 ? distortionData[4] : 0));
        } catch (RuntimeException error) {
            return Optional.empty();
        }
    }

    private static boolean hasExpectedData(JsonMatOfDouble matrix) {
        return matrix != null
                && matrix.rows > 0
                && matrix.cols > 0
                && matrix.data != null
                && (long) matrix.rows * matrix.cols == matrix.data.length;
    }

    private static boolean isSupportedDistortionCount(int count) {
        return count == 4 || count == 5;
    }

    private static boolean hasUnsupportedCudaDistortion(CameraCalibrationCoefficients calibration) {
        if (calibration == null || !hasExpectedData(calibration.distCoeffs)) {
            return false;
        }

        try {
            Mat distortion = calibration.getDistCoeffsMat();
            return (distortion.rows() == 1 || distortion.cols() == 1)
                    && distortion.channels() == 1
                    && !isSupportedDistortionCount((int) distortion.total());
        } catch (RuntimeException error) {
            return false;
        }
    }

    private static boolean allFinite(double[] values) {
        for (double value : values) {
            if (!Double.isFinite(value)) {
                return false;
            }
        }
        return true;
    }

    @Override
    protected CVPipelineResult process(Frame frame, AprilTagPipelineSettings settings) {
        long sumPipeNanosElapsed = 0L;

        if (frame.type != FrameThresholdType.GREYSCALE) {
            // We asked for a GREYSCALE frame, but didn't get one -- best we can do is give up
            return new CVPipelineResult(frame.sequenceID, 0, 0, List.of(), frame);
        }

        List<AprilTagDetection> detections = List.of();
        List<RotatedRect> mlDetectionRois = List.of();
        boolean cudaResultWasValid = false;
        if (cudaDetectionPipe.isAvailable()) {
            var cudaResult = cudaDetectionPipe.run(frame.processedImage);
            sumPipeNanosElapsed += cudaResult.nanosElapsed;
            if (cudaDetectionPipe.isAvailable()) {
                detections = cudaResult.output;
                cudaResultWasValid = true;
            }
        }

        if (!cudaResultWasValid && settings.useMLDetection && mlHybridPipe.isAvailable()) {
            var mlResult = mlHybridPipe.run(frame);
            detections = mlResult.output.detections();
            mlDetectionRois = mlResult.output.rois();
            sumPipeNanosElapsed += mlResult.nanosElapsed;
            if (detections.isEmpty() && settings.mlFallbackToTraditional) {
                var fallbackResult = aprilTagDetectionPipe.run(frame.processedImage);
                sumPipeNanosElapsed += fallbackResult.nanosElapsed;
                detections = fallbackResult.output;
            }
        } else if (!cudaResultWasValid) {
            var tagDetectionPipeResult = aprilTagDetectionPipe.run(frame.processedImage);
            sumPipeNanosElapsed += tagDetectionPipeResult.nanosElapsed;
            detections = tagDetectionPipeResult.output;
        }
        List<AprilTagDetection> usedDetections = new ArrayList<>();
        List<TrackedTarget> targetList = new ArrayList<>();

        // Filter out detections based on pipeline settings
        for (AprilTagDetection detection : detections) {
            // TODO this should be in a pipe, not in the top level here (Matt)
            if (detection.getDecisionMargin() < settings.decisionMargin) continue;
            if (detection.getHamming() > settings.hammingDist) continue;

            usedDetections.add(detection);

            // Populate target list for multitag
            // (TODO: Address circular dependencies. Multitag only requires corners and IDs, this should
            // not be necessary.)
            TrackedTarget target =
                    new TrackedTarget(
                            detection,
                            null,
                            new TargetCalculationParameters(
                                    false, null, null, null, null, currentTargetCalculationProperties));

            targetList.add(target);
        }

        // Do multi-tag pose estimation
        Optional<MultiTargetPNPResult> multiTagResult = Optional.empty();
        if (settings.solvePNPEnabled && currentPoseProcessingAllowed && settings.doMultiTarget) {
            var multiTagOutput = multiTagPNPPipe.run(targetList);
            sumPipeNanosElapsed += multiTagOutput.nanosElapsed;
            multiTagResult = multiTagOutput.output;
        }

        // Do single-tag pose estimation
        if (settings.solvePNPEnabled && currentPoseProcessingAllowed) {
            // Clear target list that was used for multitag so we can add target transforms
            targetList.clear();
            // TODO global state again ew
            var atfl = ConfigManager.getInstance().getConfig().getApriltagFieldLayout();

            for (AprilTagDetection detection : usedDetections) {
                AprilTagPoseEstimate tagPoseEstimate = null;
                // Do single-tag estimation when "always enabled" or if a tag was not used for multitag
                if (settings.doSingleTargetAlways
                        || !(multiTagResult.isPresent()
                                && multiTagResult.get().fiducialIDsUsed.contains((short) detection.getId()))) {
                    var poseResult = singleTagPoseEstimatorPipe.run(detection);
                    sumPipeNanosElapsed += poseResult.nanosElapsed;
                    tagPoseEstimate = poseResult.output;
                }

                // If single-tag estimation was not done, this is a multi-target tag from the layout
                if (tagPoseEstimate == null && multiTagResult.isPresent()) {
                    // compute this tag's camera-to-tag transform using the multitag result
                    var tagPose = atfl.getTagPose(detection.getId());
                    if (tagPose.isPresent()) {
                        var camToTag =
                                new Transform3d(
                                        new Pose3d().plus(multiTagResult.get().estimatedPose.best), tagPose.get());
                        // match expected AprilTag coordinate system
                        camToTag =
                                CoordinateSystem.convert(camToTag, CoordinateSystem.NWU(), CoordinateSystem.EDN());
                        // (AprilTag expects Z axis going into tag)
                        camToTag =
                                new Transform3d(
                                        camToTag.getTranslation(),
                                        new Rotation3d(0, Math.PI, 0).plus(camToTag.getRotation()));
                        tagPoseEstimate = new AprilTagPoseEstimate(camToTag, camToTag, 0, 0);
                    }
                }

                // populate the target list
                // Challenge here is that TrackedTarget functions with OpenCV Contour
                TrackedTarget target =
                        new TrackedTarget(
                                detection,
                                tagPoseEstimate,
                                new TargetCalculationParameters(
                                        false, null, null, null, null, currentTargetCalculationProperties));

                var correctedBestPose =
                        MathUtils.convertOpenCVtoPhotonTransform(target.getBestCameraToTarget3d());
                var correctedAltPose =
                        MathUtils.convertOpenCVtoPhotonTransform(target.getAltCameraToTarget3d());

                target.setBestCameraToTarget3d(
                        new Transform3d(correctedBestPose.getTranslation(), correctedBestPose.getRotation()));
                target.setAltCameraToTarget3d(
                        new Transform3d(correctedAltPose.getTranslation(), correctedAltPose.getRotation()));

                targetList.add(target);
            }
        }

        if (targetList.size() > Packet.MAX_ARRAY_LEN) {
            logger.error(
                    "We have " + targetList.size() + " targets! Arbitrarily dropping some on the floor");
            targetList = targetList.subList(0, Packet.MAX_ARRAY_LEN);
        }

        var fpsResult = calculateFPSPipe.run(null);
        var fps = fpsResult.output;

        return new CVPipelineResult(
                frame.sequenceID,
                sumPipeNanosElapsed,
                fps,
                targetList,
                multiTagResult,
                frame,
                List.of(),
                mlDetectionRois);
    }

    @Override
    public void release() {
        aprilTagDetectionPipe.release();
        cudaDetectionPipe.release();
        mlHybridPipe.release();
        singleTagPoseEstimatorPipe.release();
        super.release();
    }
}
