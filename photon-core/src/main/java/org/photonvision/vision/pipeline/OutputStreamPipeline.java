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

import edu.wpi.first.math.Pair;
import java.util.List;
import org.opencv.core.RotatedRect;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.opencv.DualOffsetValues;
import org.photonvision.vision.pipe.impl.*;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.photonvision.vision.target.TrackedTarget;

/**
 * This is a "fake" pipeline that is just used to move identical pipe sets out of real pipelines. It
 * shall not get its settings saved, nor shall it be managed by PipelineManager
 */
public class OutputStreamPipeline {
    private static final Logger logger =
            new Logger(OutputStreamPipeline.class, LogGroup.VisionModule);
    private static final boolean PROFILE_ENABLED =
            Boolean.getBoolean("photonvision.profile.apriltag");
    private static final int PROFILE_PERIOD_FRAMES =
            Math.max(1, Integer.getInteger("photonvision.profile.period", 60));

    private final OutputMatPipe outputMatPipe = new OutputMatPipe();
    private final Draw2dCrosshairPipe draw2dCrosshairPipe = new Draw2dCrosshairPipe();
    private final Draw2dTargetsPipe draw2dTargetsPipe = new Draw2dTargetsPipe();
    private final Draw3dTargetsPipe draw3dTargetsPipe = new Draw3dTargetsPipe();
    private final Draw2dAprilTagsPipe draw2dAprilTagsPipe = new Draw2dAprilTagsPipe();
    private final DrawMLROIPipe drawMLROIPipe = new DrawMLROIPipe();
    private final Draw3dAprilTagsPipe draw3dAprilTagsPipe = new Draw3dAprilTagsPipe();
    private final DrawCalibrationPipe drawCalibrationPipe = new DrawCalibrationPipe();

    private final Draw2dArucoPipe draw2dArucoPipe = new Draw2dArucoPipe();
    private final Draw3dArucoPipe draw3dArucoPipe = new Draw3dArucoPipe();
    private final CalculateFPSPipe calculateFPSPipe = new CalculateFPSPipe();
    private final ResizeImagePipe resizeImagePipe = new ResizeImagePipe();

    private final long[] pipeProfileNanos = new long[12];
    private int profileFrames;
    private long profileTotalNanos;
    private long profileResizeInputNanos;
    private long profileResizeOutputNanos;
    private long profileOutputMatNanos;
    private long profileCrosshairNanos;
    private long profileDrawTargetsNanos;
    private long profileDrawMlRoiNanos;
    private int profileTargets;
    private int profileRois;

    private static double averageMillis(long nanos, int frames) {
        return nanos / 1e6 / Math.max(1, frames);
    }

    private static double averageCount(int count, int frames) {
        return (double) count / Math.max(1, frames);
    }

    private void recordProfile(
            long totalNanos,
            long drawMlRoiNanos,
            int targetCount,
            int roiCount,
            AprilTagPipelineSettings aprilTagSettings) {
        if (!PROFILE_ENABLED || aprilTagSettings == null) {
            return;
        }

        profileFrames++;
        profileTotalNanos += totalNanos;
        profileResizeInputNanos += pipeProfileNanos[0];
        profileResizeOutputNanos += pipeProfileNanos[1];
        profileOutputMatNanos += pipeProfileNanos[2];
        profileCrosshairNanos += pipeProfileNanos[3] + pipeProfileNanos[4];
        profileDrawTargetsNanos += pipeProfileNanos[5] + pipeProfileNanos[7] + pipeProfileNanos[8];
        profileDrawMlRoiNanos += drawMlRoiNanos;
        profileTargets += targetCount;
        profileRois += roiCount;

        if (profileFrames < PROFILE_PERIOD_FRAMES) {
            return;
        }

        logger.info(
                String.format(
                        "PVPROFILE stream frames=%d totalMs=%.2f resizeInMs=%.2f resizeOutMs=%.2f outputMatMs=%.2f crosshairMs=%.3f drawTargetsMs=%.2f drawMlRoiMs=%.3f targets=%.2f rois=%.2f outputDraw=%s streamDivisor=%s",
                        profileFrames,
                        averageMillis(profileTotalNanos, profileFrames),
                        averageMillis(profileResizeInputNanos, profileFrames),
                        averageMillis(profileResizeOutputNanos, profileFrames),
                        averageMillis(profileOutputMatNanos, profileFrames),
                        averageMillis(profileCrosshairNanos, profileFrames),
                        averageMillis(profileDrawTargetsNanos, profileFrames),
                        averageMillis(profileDrawMlRoiNanos, profileFrames),
                        averageCount(profileTargets, profileFrames),
                        averageCount(profileRois, profileFrames),
                        aprilTagSettings.outputShouldDraw,
                        aprilTagSettings.streamingFrameDivisor));

        profileFrames = 0;
        profileTotalNanos = 0;
        profileResizeInputNanos = 0;
        profileResizeOutputNanos = 0;
        profileOutputMatNanos = 0;
        profileCrosshairNanos = 0;
        profileDrawTargetsNanos = 0;
        profileDrawMlRoiNanos = 0;
        profileTargets = 0;
        profileRois = 0;
    }

    protected void setPipeParams(
            FrameStaticProperties frameStaticProperties, AdvancedPipelineSettings settings) {
        var dualOffsetValues =
                new DualOffsetValues(
                        settings.offsetDualPointA,
                        settings.offsetDualPointAArea,
                        settings.offsetDualPointB,
                        settings.offsetDualPointBArea);

        draw2dTargetsPipe.setParams(
                new Draw2dTargetsPipe.Draw2dTargetsParams(
                        settings.outputShouldDraw,
                        settings.outputMaximumTargets,
                        settings.streamingFrameDivisor));

        draw2dAprilTagsPipe.setParams(
                new Draw2dAprilTagsPipe.Draw2dAprilTagsParams(
                        settings.outputShouldDraw,
                        settings.outputMaximumTargets,
                        settings.streamingFrameDivisor));

        if (settings instanceof AprilTagPipelineSettings atSettings) {
            drawMLROIPipe.setParams(
                    new DrawMLROIPipe.DrawMLROIParams(
                            settings.outputShouldDraw,
                            atSettings.showDetectionBoxes,
                            settings.streamingFrameDivisor));
        }

        draw2dArucoPipe.setParams(
                new Draw2dArucoPipe.Draw2dArucoParams(
                        settings.outputShouldDraw,
                        settings.outputMaximumTargets,
                        settings.streamingFrameDivisor));

        draw2dCrosshairPipe.setParams(
                new Draw2dCrosshairPipe.Draw2dCrosshairParams(
                        settings.outputShouldDraw,
                        settings.offsetRobotOffsetMode,
                        settings.offsetSinglePoint,
                        dualOffsetValues,
                        frameStaticProperties,
                        settings.streamingFrameDivisor,
                        settings.inputImageRotationMode));

        draw3dTargetsPipe.setParams(
                new Draw3dTargetsPipe.Draw3dContoursParams(
                        settings.outputShouldDraw,
                        frameStaticProperties.cameraCalibration,
                        settings.targetModel,
                        settings.streamingFrameDivisor));

        draw3dAprilTagsPipe.setParams(
                new Draw3dAprilTagsPipe.Draw3dAprilTagsParams(
                        settings.outputShouldDraw,
                        frameStaticProperties.cameraCalibration,
                        settings.targetModel,
                        settings.streamingFrameDivisor));

        draw3dArucoPipe.setParams(
                new Draw3dArucoPipe.Draw3dArucoParams(
                        settings.outputShouldDraw,
                        frameStaticProperties.cameraCalibration,
                        settings.targetModel,
                        settings.streamingFrameDivisor));

        resizeImagePipe.setParams(
                new ResizeImagePipe.ResizeImageParams(settings.streamingFrameDivisor));

        if (settings instanceof Calibration3dPipelineSettings pipelineSettings) {
            drawCalibrationPipe.setParams(
                    new DrawCalibrationPipe.DrawCalibrationPipeParams(
                            pipelineSettings.streamingFrameDivisor, pipelineSettings.drawAllSnapshots));
        }
    }

    public CVPipelineResult process(
            Frame inputAndOutputFrame,
            AdvancedPipelineSettings settings,
            List<TrackedTarget> targetsToDraw) {
        return process(inputAndOutputFrame, settings, targetsToDraw, List.of());
    }

    public CVPipelineResult process(
            Frame inputAndOutputFrame,
            AdvancedPipelineSettings settings,
            List<TrackedTarget> targetsToDraw,
            List<RotatedRect> mlDetectionRois) {
        setPipeParams(inputAndOutputFrame.frameStaticProperties, settings);
        var inMat = inputAndOutputFrame.colorImage.getMat();
        var outMat = inputAndOutputFrame.processedImage.getMat();

        long sumPipeNanosElapsed = 0L;
        long drawMlRoiNanos = 0L;

        // Resize both in place before doing any conversion
        boolean inEmpty = inMat.empty();
        if (!inEmpty)
            sumPipeNanosElapsed += pipeProfileNanos[0] = resizeImagePipe.run(inMat).nanosElapsed;

        boolean outEmpty = outMat.empty();
        if (!outEmpty)
            sumPipeNanosElapsed += pipeProfileNanos[1] = resizeImagePipe.run(outMat).nanosElapsed;

        // Only attempt drawing on a non-empty frame
        if (!outEmpty) {
            // Convert single-channel HSV output mat to 3-channel BGR in preparation for streaming
            if (outMat.channels() == 1) {
                var outputMatPipeResult = outputMatPipe.run(outMat);
                sumPipeNanosElapsed += pipeProfileNanos[2] = outputMatPipeResult.nanosElapsed;
            } else {
                pipeProfileNanos[2] = 0;
            }

            // Draw 2D Crosshair on output
            var draw2dCrosshairResultOnInput = draw2dCrosshairPipe.run(Pair.of(inMat, targetsToDraw));
            sumPipeNanosElapsed += pipeProfileNanos[3] = draw2dCrosshairResultOnInput.nanosElapsed;

            if (!(settings instanceof AprilTagPipelineSettings)
                    && !(settings instanceof ArucoPipelineSettings)
                    && !(settings instanceof Calibration3dPipelineSettings)) {
                // If we're processing anything other than AprilTags or Aruco targets
                var draw2dCrosshairResultOnOutput = draw2dCrosshairPipe.run(Pair.of(outMat, targetsToDraw));
                sumPipeNanosElapsed += pipeProfileNanos[4] = draw2dCrosshairResultOnOutput.nanosElapsed;

                if (settings.solvePNPEnabled) {
                    // Draw 3D Targets on input and output if possible
                    pipeProfileNanos[5] = 0;
                    pipeProfileNanos[6] = 0;
                    pipeProfileNanos[7] = 0;

                    var drawOnOutputResult = draw3dTargetsPipe.run(Pair.of(outMat, targetsToDraw));
                    sumPipeNanosElapsed += pipeProfileNanos[8] = drawOnOutputResult.nanosElapsed;
                } else {
                    // Only draw 2d targets
                    pipeProfileNanos[5] = 0;

                    var draw2dTargetsOnOutput = draw2dTargetsPipe.run(Pair.of(outMat, targetsToDraw));
                    sumPipeNanosElapsed += pipeProfileNanos[6] = draw2dTargetsOnOutput.nanosElapsed;

                    pipeProfileNanos[7] = 0;
                    pipeProfileNanos[8] = 0;
                }
            } else if (settings instanceof Calibration3dPipelineSettings) {
                pipeProfileNanos[5] = 0;
                pipeProfileNanos[6] = 0;

                var drawOnInputResult = drawCalibrationPipe.run(Pair.of(outMat, targetsToDraw));
                sumPipeNanosElapsed += pipeProfileNanos[7] = drawOnInputResult.nanosElapsed;

                pipeProfileNanos[8] = 0;
            } else if (settings instanceof AprilTagPipelineSettings) {
                // If we are doing apriltags...
                var drawMlRoiResult = drawMLROIPipe.run(Pair.of(outMat, mlDetectionRois));
                drawMlRoiNanos = drawMlRoiResult.nanosElapsed;
                sumPipeNanosElapsed += drawMlRoiNanos;

                if (settings.solvePNPEnabled) {
                    // Draw 3d Apriltag markers (camera is calibrated and running in 3d mode)
                    pipeProfileNanos[5] = 0;
                    pipeProfileNanos[6] = 0;

                    var drawOnInputResult = draw3dAprilTagsPipe.run(Pair.of(outMat, targetsToDraw));
                    sumPipeNanosElapsed += pipeProfileNanos[7] = drawOnInputResult.nanosElapsed;

                    pipeProfileNanos[8] = 0;

                } else {
                    // Draw 2d apriltag markers
                    var draw2dTargetsOnInput = draw2dAprilTagsPipe.run(Pair.of(outMat, targetsToDraw));
                    sumPipeNanosElapsed += pipeProfileNanos[5] = draw2dTargetsOnInput.nanosElapsed;

                    pipeProfileNanos[6] = 0;
                    pipeProfileNanos[7] = 0;
                    pipeProfileNanos[8] = 0;
                }
            } else if (settings instanceof ArucoPipelineSettings) {
                if (settings.solvePNPEnabled) {
                    // Draw 3d aruco markers (camera is calibrated and running in 3d mode)
                    pipeProfileNanos[5] = 0;
                    pipeProfileNanos[6] = 0;

                    var drawOnInputResult = draw3dArucoPipe.run(Pair.of(outMat, targetsToDraw));
                    sumPipeNanosElapsed += pipeProfileNanos[7] = drawOnInputResult.nanosElapsed;

                    pipeProfileNanos[8] = 0;

                } else {
                    // Draw 2d aruco markers
                    var draw2dTargetsOnInput = draw2dArucoPipe.run(Pair.of(outMat, targetsToDraw));
                    sumPipeNanosElapsed += pipeProfileNanos[5] = draw2dTargetsOnInput.nanosElapsed;

                    pipeProfileNanos[6] = 0;
                    pipeProfileNanos[7] = 0;
                    pipeProfileNanos[8] = 0;
                }
            }
        }

        var fpsResult = calculateFPSPipe.run(null);
        var fps = fpsResult.output;
        recordProfile(
                sumPipeNanosElapsed,
                drawMlRoiNanos,
                targetsToDraw.size(),
                mlDetectionRois.size(),
                settings instanceof AprilTagPipelineSettings atSettings ? atSettings : null);

        return new CVPipelineResult(
                inputAndOutputFrame.sequenceID,
                sumPipeNanosElapsed,
                fps, // Unused but here just in case
                targetsToDraw,
                inputAndOutputFrame);
    }
}
