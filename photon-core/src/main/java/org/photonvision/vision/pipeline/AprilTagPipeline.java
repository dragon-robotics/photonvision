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
import org.opencv.core.RotatedRect;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.configuration.NeuralNetworkModelManager;
import org.photonvision.common.dataflow.structures.Packet;
import org.photonvision.common.hardware.Platform;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.math.MathUtils;
import org.photonvision.estimation.TargetModel;
import org.photonvision.targeting.MultiTargetPNPResult;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.objects.Model;
import org.photonvision.vision.pipe.CVPipe.CVPipeResult;
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
    private static final int RUBIK_ML_MAX_DECODE_THREADS = 4;
    private static final int RUBIK_ML_MAX_ATR_TARGET_DIMENSION = 144;
    private static final boolean PROFILE_ENABLED =
            Boolean.getBoolean("photonvision.profile.apriltag");
    private static final int PROFILE_PERIOD_FRAMES =
            Math.max(1, Integer.getInteger("photonvision.profile.period", 60));
    private static final int PROFILE_ML_REFRESH_INTERVAL_FRAMES =
            Math.max(1, Integer.getInteger("photonvision.profile.mlRefreshInterval", 1));
    private static final int PROFILE_ML_DECODE_THREADS_OVERRIDE =
            Integer.getInteger("photonvision.profile.mlDecodeThreads", -1);

    private final AprilTagDetectionPipe aprilTagDetectionPipe = new AprilTagDetectionPipe();
    private final AprilTagPoseEstimatorPipe singleTagPoseEstimatorPipe =
            new AprilTagPoseEstimatorPipe();
    private final MultiTargetPNPPipe multiTagPNPPipe = new MultiTargetPNPPipe();
    private final CalculateFPSPipe calculateFPSPipe = new CalculateFPSPipe();
    private final AprilTagMLHybridPipe mlHybridPipe = new AprilTagMLHybridPipe();
    private boolean mlAvailable = false;
    private boolean mlWasAvailable = false;
    private int profileFrames;
    private int profileMlFrames;
    private long profileProcessNanos;
    private long profileDetectionNanos;
    private long profileMlRoiNanos;
    private long profileMlRoiExpansionNanos;
    private long profileMlDecodeNanos;
    private long profileFilterTargetNanos;
    private long profileMultiTagNanos;
    private long profileSingleStageNanos;
    private long profileSinglePoseNanos;
    private int profileDetections;
    private int profileUsedDetections;
    private int profileTargets;
    private int profileMlRawRois;
    private int profileMlExpandedRois;
    private int profileMlFreshFrames;
    private int profileMlCachedFrames;
    private int profileMlForcedRefreshFrames;
    private int profileMlDecodeDropFrames;
    private double profileFps;

    private static final FrameThresholdType PROCESSING_TYPE = FrameThresholdType.GREYSCALE;

    public AprilTagPipeline() {
        super(PROCESSING_TYPE);
        settings = new AprilTagPipelineSettings();
    }

    public AprilTagPipeline(AprilTagPipelineSettings settings) {
        super(PROCESSING_TYPE);
        this.settings = settings;
    }

    private static double averageMillis(long nanos, int frames) {
        return nanos / 1e6 / Math.max(1, frames);
    }

    private static double averageCount(int count, int frames) {
        return (double) count / Math.max(1, frames);
    }

    private void recordProfile(
            boolean usedMl,
            long processNanos,
            long detectionNanos,
            long mlRoiNanos,
            long mlRoiExpansionNanos,
            long mlDecodeNanos,
            long filterTargetNanos,
            long multiTagNanos,
            long singleStageNanos,
            long singlePoseNanos,
            int detectionCount,
            int usedDetectionCount,
            int targetCount,
            int rawRoiCount,
            int expandedRoiCount,
            boolean freshMlFrame,
            boolean usedCachedRois,
            boolean forcedRefresh,
            boolean decodeDrop,
            double fps) {
        if (!PROFILE_ENABLED) {
            return;
        }

        profileFrames++;
        if (usedMl) {
            profileMlFrames++;
        }
        profileProcessNanos += processNanos;
        profileDetectionNanos += detectionNanos;
        profileMlRoiNanos += mlRoiNanos;
        profileMlRoiExpansionNanos += mlRoiExpansionNanos;
        profileMlDecodeNanos += mlDecodeNanos;
        profileFilterTargetNanos += filterTargetNanos;
        profileMultiTagNanos += multiTagNanos;
        profileSingleStageNanos += singleStageNanos;
        profileSinglePoseNanos += singlePoseNanos;
        profileDetections += detectionCount;
        profileUsedDetections += usedDetectionCount;
        profileTargets += targetCount;
        profileMlRawRois += rawRoiCount;
        profileMlExpandedRois += expandedRoiCount;
        if (freshMlFrame) {
            profileMlFreshFrames++;
        }
        if (usedCachedRois) {
            profileMlCachedFrames++;
        }
        if (forcedRefresh) {
            profileMlForcedRefreshFrames++;
        }
        if (decodeDrop) {
            profileMlDecodeDropFrames++;
        }
        profileFps += fps;

        if (profileFrames < PROFILE_PERIOD_FRAMES) {
            return;
        }

        logger.info(
                String.format(
                        "PVPROFILE apriltag frames=%d mlFrames=%d mlRefreshInterval=%d mlDecodeThreads=%d mlFreshFrames=%d mlCachedFrames=%d mlForcedRefreshFrames=%d mlDecodeDropFrames=%d avgFps=%.2f processMs=%.2f detectMs=%.2f mlRoiMs=%.2f mlRoiExpandMs=%.3f mlDecodeMs=%.2f filterTargetMs=%.3f multiTagMs=%.2f singleStageMs=%.2f singlePoseMs=%.2f detections=%.2f usedDetections=%.2f targets=%.2f rawRois=%.2f expandedRois=%.2f",
                        profileFrames,
                        profileMlFrames,
                        PROFILE_ML_REFRESH_INTERVAL_FRAMES,
                        getEffectiveMlDecodeThreads(
                                settings.threads,
                                Platform.getCurrentPlatform(),
                                PROFILE_ML_DECODE_THREADS_OVERRIDE),
                        profileMlFreshFrames,
                        profileMlCachedFrames,
                        profileMlForcedRefreshFrames,
                        profileMlDecodeDropFrames,
                        profileFps / profileFrames,
                        averageMillis(profileProcessNanos, profileFrames),
                        averageMillis(profileDetectionNanos, profileFrames),
                        averageMillis(profileMlRoiNanos, profileFrames),
                        averageMillis(profileMlRoiExpansionNanos, profileFrames),
                        averageMillis(profileMlDecodeNanos, profileFrames),
                        averageMillis(profileFilterTargetNanos, profileFrames),
                        averageMillis(profileMultiTagNanos, profileFrames),
                        averageMillis(profileSingleStageNanos, profileFrames),
                        averageMillis(profileSinglePoseNanos, profileFrames),
                        averageCount(profileDetections, profileFrames),
                        averageCount(profileUsedDetections, profileFrames),
                        averageCount(profileTargets, profileFrames),
                        averageCount(profileMlRawRois, profileFrames),
                        averageCount(profileMlExpandedRois, profileFrames)));

        profileFrames = 0;
        profileMlFrames = 0;
        profileProcessNanos = 0;
        profileDetectionNanos = 0;
        profileMlRoiNanos = 0;
        profileMlRoiExpansionNanos = 0;
        profileMlDecodeNanos = 0;
        profileFilterTargetNanos = 0;
        profileMultiTagNanos = 0;
        profileSingleStageNanos = 0;
        profileSinglePoseNanos = 0;
        profileDetections = 0;
        profileUsedDetections = 0;
        profileTargets = 0;
        profileMlRawRois = 0;
        profileMlExpandedRois = 0;
        profileMlFreshFrames = 0;
        profileMlCachedFrames = 0;
        profileMlForcedRefreshFrames = 0;
        profileMlDecodeDropFrames = 0;
        profileFps = 0;
    }

    static int getEffectiveMlDecodeThreads(int requestedThreads, Platform platform) {
        return getEffectiveMlDecodeThreads(requestedThreads, platform, -1);
    }

    static int getEffectiveMlDecodeThreads(
            int requestedThreads, Platform platform, int overrideThreads) {
        if (overrideThreads > 0) {
            requestedThreads = overrideThreads;
        }
        int sanitizedThreads = Math.max(1, requestedThreads);
        if (platform == Platform.LINUX_QCS6490) {
            return Math.min(sanitizedThreads, RUBIK_ML_MAX_DECODE_THREADS);
        }
        return sanitizedThreads;
    }

    static int getEffectiveMlAtrTargetDimension(int requestedDimension, Platform platform) {
        int sanitizedDimension = Math.max(1, requestedDimension);
        if (platform == Platform.LINUX_QCS6490) {
            return Math.min(sanitizedDimension, RUBIK_ML_MAX_ATR_TARGET_DIMENSION);
        }
        return sanitizedDimension;
    }

    @Override
    protected void setPipeParamsImpl() {
        // Sanitize thread count - not supported to have fewer than 1 threads
        settings.threads = Math.max(1, settings.threads);
        var currentPlatform = Platform.getCurrentPlatform();
        if (settings.useMLDetection && currentPlatform == Platform.LINUX_QCS6490) {
            settings.threads =
                    getEffectiveMlDecodeThreads(
                            settings.threads, currentPlatform, PROFILE_ML_DECODE_THREADS_OVERRIDE);
            settings.atrTargetDimension =
                    getEffectiveMlAtrTargetDimension(settings.atrTargetDimension, currentPlatform);
        }

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
            boolean platformOk = currentPlatform == Platform.LINUX_QCS6490;
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
                decodeParams.detectorConfig.numThreads =
                        getEffectiveMlDecodeThreads(
                                settings.threads, currentPlatform, PROFILE_ML_DECODE_THREADS_OVERRIDE);
                decodeParams.detectorConfig.refineEdges = settings.refineEdges;
                decodeParams.detectorConfig.quadDecimate = 1;
                decodeParams.detectorConfig.quadSigma = (float) settings.blur;
                decodeParams.quadParams = quadParams;
                decodeParams.atrEnabled = settings.atrEnabled;
                decodeParams.atrTargetDimension =
                        getEffectiveMlAtrTargetDimension(settings.atrTargetDimension, currentPlatform);
                decodeParams.atrMinScaleFactor = settings.atrMinScaleFactor;

                mlHybridPipe.setParams(
                        new AprilTagMLHybridPipe.Params(
                                detectionParams,
                                decodeParams,
                                settings.mlRoiPaddingPixels,
                                PROFILE_ML_REFRESH_INTERVAL_FRAMES));
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

        if (frameStaticProperties.cameraCalibration != null) {
            var cameraMatrix = frameStaticProperties.cameraCalibration.getCameraIntrinsicsMat();
            if (cameraMatrix != null && cameraMatrix.rows() > 0) {
                var cx = cameraMatrix.get(0, 2)[0];
                var cy = cameraMatrix.get(1, 2)[0];
                var fx = cameraMatrix.get(0, 0)[0];
                var fy = cameraMatrix.get(1, 1)[0];

                singleTagPoseEstimatorPipe.setParams(
                        new AprilTagPoseEstimatorPipeParams(
                                new Config(tagWidth, fx, fy, cx, cy),
                                frameStaticProperties.cameraCalibration,
                                settings.numIterations));

                // TODO global state ew
                var atfl = ConfigManager.getInstance().getConfig().getApriltagFieldLayout();
                multiTagPNPPipe.setParams(
                        new MultiTargetPNPPipeParams(frameStaticProperties.cameraCalibration, atfl, tagModel));
            }
        }
    }

    @Override
    protected CVPipelineResult process(Frame frame, AprilTagPipelineSettings settings) {
        long processStartNanos = System.nanoTime();
        long sumPipeNanosElapsed = 0L;

        if (frame.type != FrameThresholdType.GREYSCALE) {
            // We asked for a GREYSCALE frame, but didn't get one -- best we can do is give up
            return new CVPipelineResult(frame.sequenceID, 0, 0, List.of(), frame);
        }

        List<AprilTagDetection> detections;
        long detectionNanos;
        List<RotatedRect> mlDetectionRois = List.of();
        boolean usedMl = settings.useMLDetection && mlAvailable;
        long mlRoiNanos = 0;
        long mlRoiExpansionNanos = 0;
        long mlDecodeNanos = 0;
        int rawRoiCount = 0;
        int expandedRoiCount = 0;

        if (usedMl) {
            var hybridResult = mlHybridPipe.run(frame);
            detections = hybridResult.output.detections();
            detectionNanos = hybridResult.nanosElapsed;
            mlDetectionRois = hybridResult.output.rois();
            mlRoiNanos = mlHybridPipe.getLastRoiDetectionNanos();
            mlRoiExpansionNanos = mlHybridPipe.getLastRoiExpansionNanos();
            mlDecodeNanos = mlHybridPipe.getLastRoiDecodeNanos();
            rawRoiCount = mlHybridPipe.getLastRawRoiCount();
            expandedRoiCount = mlHybridPipe.getLastExpandedRoiCount();
        } else {
            CVPipeResult<List<AprilTagDetection>> tagDetectionPipeResult =
                    aprilTagDetectionPipe.run(frame.processedImage);
            detections = tagDetectionPipeResult.output;
            detectionNanos = tagDetectionPipeResult.nanosElapsed;
        }
        sumPipeNanosElapsed += detectionNanos;
        List<AprilTagDetection> usedDetections = new ArrayList<>();
        List<TrackedTarget> targetList = new ArrayList<>();

        // Filter out detections based on pipeline settings
        long filterTargetStartNanos = System.nanoTime();
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
                                    false, null, null, null, null, frameStaticProperties));

            targetList.add(target);
        }
        long filterTargetNanos = System.nanoTime() - filterTargetStartNanos;

        // Do multi-tag pose estimation
        Optional<MultiTargetPNPResult> multiTagResult = Optional.empty();
        long multiTagNanos = 0;
        if (settings.solvePNPEnabled && settings.doMultiTarget) {
            var multiTagOutput = multiTagPNPPipe.run(targetList);
            sumPipeNanosElapsed += multiTagOutput.nanosElapsed;
            multiTagNanos = multiTagOutput.nanosElapsed;
            multiTagResult = multiTagOutput.output;
        }

        // Do single-tag pose estimation
        long singleStageNanos = 0;
        long singlePoseNanos = 0;
        if (settings.solvePNPEnabled) {
            long singleStageStartNanos = System.nanoTime();
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
                    singlePoseNanos += poseResult.nanosElapsed;
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
                                        false, null, null, null, null, frameStaticProperties));

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
            singleStageNanos = System.nanoTime() - singleStageStartNanos;
        }

        if (targetList.size() > Packet.MAX_ARRAY_LEN) {
            logger.error(
                    "We have " + targetList.size() + " targets! Arbitrarily dropping some on the floor");
            targetList = targetList.subList(0, Packet.MAX_ARRAY_LEN);
        }

        var fpsResult = calculateFPSPipe.run(null);
        var fps = fpsResult.output;
        long processNanos = System.nanoTime() - processStartNanos;

        recordProfile(
                usedMl,
                processNanos,
                detectionNanos,
                mlRoiNanos,
                mlRoiExpansionNanos,
                mlDecodeNanos,
                filterTargetNanos,
                multiTagNanos,
                singleStageNanos,
                singlePoseNanos,
                detections.size(),
                usedDetections.size(),
                targetList.size(),
                rawRoiCount,
                expandedRoiCount,
                usedMl && mlHybridPipe.getLastFreshMlFrame(),
                usedMl && mlHybridPipe.getLastUsedCachedRois(),
                usedMl && mlHybridPipe.getLastForcedRefresh(),
                usedMl && mlHybridPipe.getLastDecodeDrop(),
                fps);

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
        mlHybridPipe.release();
        singleTagPoseEstimatorPipe.release();
        super.release();
    }
}
