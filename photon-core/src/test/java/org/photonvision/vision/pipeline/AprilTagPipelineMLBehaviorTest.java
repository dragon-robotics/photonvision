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

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.ObjectMapper;
import edu.wpi.first.apriltag.AprilTagDetection;
import edu.wpi.first.apriltag.AprilTagPoseEstimate;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Queue;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Family;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Version;
import org.photonvision.common.configuration.NeuralNetworkModelsSettings.ModelProperties;
import org.photonvision.targeting.MultiTargetPNPResult;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.calibration.CameraCalibrationCoefficients;
import org.photonvision.vision.calibration.CameraLensModel;
import org.photonvision.vision.calibration.JsonMatOfDouble;
import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.objects.Model;
import org.photonvision.vision.objects.NullModel;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipe.CVPipe.CVPipeResult;
import org.photonvision.vision.pipe.impl.AprilTagDetectionCudaPipe;
import org.photonvision.vision.pipe.impl.AprilTagDetectionCudaPipe.AprilTagDetectionCudaPipeParams;
import org.photonvision.vision.pipe.impl.AprilTagDetectionCudaPipe.Calibration;
import org.photonvision.vision.pipe.impl.AprilTagDetectionPipe;
import org.photonvision.vision.pipe.impl.AprilTagMLHybridPipe;
import org.photonvision.vision.pipe.impl.AprilTagPoseEstimatorPipe;
import org.photonvision.vision.pipe.impl.AprilTagPoseEstimatorPipe.AprilTagPoseEstimatorPipeParams;
import org.photonvision.vision.pipe.impl.CalculateFPSPipe;
import org.photonvision.vision.pipe.impl.MLDetectionResult;
import org.photonvision.vision.pipe.impl.MultiTargetPNPPipe;
import org.photonvision.vision.pipe.impl.MultiTargetPNPPipe.MultiTargetPNPPipeParams;
import org.photonvision.vision.target.TrackedTarget;

public class AprilTagPipelineMLBehaviorTest {
    private static final Calibration CUDA_IDENTITY_CALIBRATION =
            new Calibration(1, 1, 0, 0, 0, 0, 0, 0, 0);

    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
        ConfigManager.getInstance().load();
    }

    @Test
    public void mlDisabledPreservesTraditionalBehavior() {
        var traditionalPipe = new FakeAprilTagDetectionPipe(List.of(makeDetection(1)));
        var mlPipe = new FakeAprilTagMLHybridPipe();
        var pipeline =
                new TestAprilTagPipeline(
                        traditionalPipe,
                        mlPipe,
                        new AprilTagPoseEstimatorPipe(),
                        new MultiTargetPNPPipe(),
                        new CalculateFPSPipe(),
                        Optional.empty());
        pipeline.getSettings().useMLDetection = false;
        pipeline.getSettings().solvePNPEnabled = false;

        var frame = makeFrame();
        var result = pipeline.run(frame, QuirkyCamera.DefaultCamera);

        assertEquals(1, traditionalPipe.runCount);
        assertEquals(0, mlPipe.runCount);
        assertEquals(1, result.targets.size());
        assertEquals(1, result.targets.get(0).getFiducialId());
        result.release();
        frame.release();
        pipeline.release();
    }

    @Test
    public void mlUnavailablePreservesTraditionalBehavior() {
        var traditionalPipe = new FakeAprilTagDetectionPipe(List.of(makeDetection(2)));
        var mlPipe = new FakeAprilTagMLHybridPipe();
        var pipeline =
                new TestAprilTagPipeline(
                        traditionalPipe,
                        mlPipe,
                        new AprilTagPoseEstimatorPipe(),
                        new MultiTargetPNPPipe(),
                        new CalculateFPSPipe(),
                        Optional.empty());
        pipeline.getSettings().useMLDetection = true;
        pipeline.getSettings().solvePNPEnabled = false;

        var frame = makeFrame();
        var result = pipeline.run(frame, QuirkyCamera.DefaultCamera);

        assertFalse(mlPipe.isAvailable());
        assertEquals(1, traditionalPipe.runCount);
        assertEquals(0, mlPipe.runCount);
        assertEquals(1, result.targets.size());
        assertEquals(2, result.targets.get(0).getFiducialId());
        result.release();
        frame.release();
        pipeline.release();
    }

    @Test
    public void mlEnabledWithNoDetectionsFallsBackToTraditional() {
        var traditionalPipe = new FakeAprilTagDetectionPipe(List.of(makeDetection(3)));
        var mlPipe = new FakeAprilTagMLHybridPipe();
        mlPipe.nextResult = new MLDetectionResult(List.of(), List.<RotatedRect>of(), 17);
        var pipeline =
                new TestAprilTagPipeline(
                        traditionalPipe,
                        mlPipe,
                        new AprilTagPoseEstimatorPipe(),
                        new MultiTargetPNPPipe(),
                        new CalculateFPSPipe(),
                        Optional.of(new FakeModel()));
        pipeline.getSettings().useMLDetection = true;
        pipeline.getSettings().mlFallbackToTraditional = true;
        pipeline.getSettings().solvePNPEnabled = false;

        var frame = makeFrame();
        var result = pipeline.run(frame, QuirkyCamera.DefaultCamera);

        assertTrue(mlPipe.isAvailable());
        assertEquals(1, mlPipe.runCount);
        assertEquals(1, traditionalPipe.runCount);
        assertEquals(1, result.targets.size());
        assertEquals(3, result.targets.get(0).getFiducialId());
        result.release();
        frame.release();
        pipeline.release();
    }

    @Test
    public void staleModelStateClearsAndRevertsToTraditional() {
        var traditionalPipe = new FakeAprilTagDetectionPipe(List.of(makeDetection(4)));
        var mlPipe = new FakeAprilTagMLHybridPipe();
        mlPipe.nextResult = new MLDetectionResult(List.of(), List.<RotatedRect>of(), 11);
        var pipeline =
                new TestAprilTagPipeline(
                        traditionalPipe,
                        mlPipe,
                        new AprilTagPoseEstimatorPipe(),
                        new MultiTargetPNPPipe(),
                        new CalculateFPSPipe(),
                        Optional.of(new FakeModel()),
                        Optional.empty());
        pipeline.getSettings().useMLDetection = true;
        pipeline.getSettings().mlFallbackToTraditional = false;
        pipeline.getSettings().solvePNPEnabled = false;

        var firstFrame = makeFrame();
        var firstResult = pipeline.run(firstFrame, QuirkyCamera.DefaultCamera);
        assertEquals(1, mlPipe.runCount);
        assertEquals(0, traditionalPipe.runCount);
        assertTrue(firstResult.targets.isEmpty());
        firstResult.release();
        firstFrame.release();

        var secondFrame = makeFrame();
        var secondResult = pipeline.run(secondFrame, QuirkyCamera.DefaultCamera);
        assertFalse(mlPipe.isAvailable());
        assertEquals(1, mlPipe.runCount);
        assertEquals(1, traditionalPipe.runCount);
        assertEquals(1, secondResult.targets.size());
        assertEquals(4, secondResult.targets.get(0).getFiducialId());
        secondResult.release();
        secondFrame.release();
        pipeline.release();
    }

    @Test
    public void throwingMlModelFallsBackToTraditionalDetection() {
        var traditionalPipe = new FakeAprilTagDetectionPipe(List.of(makeDetection(5)));
        var pipeline =
                new TestAprilTagPipeline(
                        traditionalPipe,
                        new AprilTagMLHybridPipe(),
                        new AprilTagPoseEstimatorPipe(),
                        new MultiTargetPNPPipe(),
                        new CalculateFPSPipe(),
                        Optional.of(new ThrowingModel()));
        pipeline.getSettings().useMLDetection = true;
        pipeline.getSettings().mlFallbackToTraditional = true;
        pipeline.getSettings().solvePNPEnabled = false;

        var frame = makeFrame();
        var result = pipeline.run(frame, QuirkyCamera.DefaultCamera);

        assertEquals(1, traditionalPipe.runCount);
        assertEquals(1, result.targets.size());
        assertEquals(5, result.targets.get(0).getFiducialId());
        result.release();
        frame.release();
        pipeline.release();
    }

    @Test
    public void cudaTagDetectionDefaultsFalseAndSerializesAsFalse() throws Exception {
        var defaults = new AprilTagPipelineSettings();
        var enabled = new AprilTagPipelineSettings();
        enabled.useCudaTagDetection = true;

        assertFalse(defaults.useCudaTagDetection);
        assertNotEquals(defaults, enabled);
        assertNotEquals(defaults.hashCode(), enabled.hashCode());
        assertTrue(
                new ObjectMapper().writeValueAsString(defaults).contains("\"useCudaTagDetection\":false"));
    }

    @Test
    public void cudaIsEnabledOnlyForRequestedTag36h11() {
        var cudaPipe = new FakeAprilTagDetectionCudaPipe();
        var pipeline =
                newCudaTestPipeline(
                        cudaPipe, new FakeAprilTagDetectionPipe(List.of()), new FakeAprilTagMLHybridPipe());
        pipeline.getSettings().useCudaTagDetection = true;
        pipeline.getSettings().decimate = 4;
        pipeline.getSettings().solvePNPEnabled = false;

        var firstFrame = makeFrame();
        var firstResult = pipeline.run(firstFrame, QuirkyCamera.DefaultCamera);
        assertEquals(List.of(true), cudaPipe.enabledTransitions);
        assertEquals(4, cudaPipe.lastParams.decimate());
        firstResult.release();
        firstFrame.release();

        pipeline.getSettings().tagFamily = AprilTagFamily.kTag16h5;
        var secondFrame = makeFrame();
        var secondResult = pipeline.run(secondFrame, QuirkyCamera.DefaultCamera);
        assertEquals(List.of(true, false), cudaPipe.enabledTransitions);
        secondResult.release();
        secondFrame.release();
        pipeline.release();
    }

    @Test
    public void nullCalibrationResetsPreviouslyValidCudaCalibration() {
        var cudaPipe = new FakeAprilTagDetectionCudaPipe();
        var pipeline =
                newCudaTestPipeline(
                        cudaPipe, new FakeAprilTagDetectionPipe(List.of()), new FakeAprilTagMLHybridPipe());
        pipeline.getSettings().useCudaTagDetection = true;
        pipeline.getSettings().solvePNPEnabled = false;
        var validCalibration = makeCalibration(validIntrinsics(), validDistortion());

        var calibratedFrame = makeFrame(validCalibration);
        var calibratedResult = pipeline.run(calibratedFrame, QuirkyCamera.DefaultCamera);
        calibratedResult.release();
        calibratedFrame.release();

        var uncalibratedFrame = makeFrame();
        var uncalibratedResult = pipeline.run(uncalibratedFrame, QuirkyCamera.DefaultCamera);

        assertEquals(
                List.of(
                        new Calibration(700, 710, 320, 240, 0.1, -0.2, 0.003, -0.004, 0.05),
                        CUDA_IDENTITY_CALIBRATION),
                cudaPipe.calibrationUpdates);
        uncalibratedResult.release();
        uncalibratedFrame.release();
        validCalibration.release();
        pipeline.release();
    }

    @Test
    public void malformedCalibrationResetsCudaAndLeavesCpuFallbackUsable() {
        var cudaPipe = new FakeAprilTagDetectionCudaPipe();
        cudaPipe.available = false;
        var cpuPipe = new FakeAprilTagDetectionPipe(List.of(makeDetection(11)));
        var pipeline = newCudaTestPipeline(cudaPipe, cpuPipe, new FakeAprilTagMLHybridPipe());
        pipeline.getSettings().useCudaTagDetection = true;
        pipeline.getSettings().useMLDetection = false;
        pipeline.getSettings().solvePNPEnabled = false;
        var malformedIntrinsics =
                makeCalibration(new JsonMatOfDouble(1, 1, new double[] {700}), validDistortion());
        var malformedDistortion =
                makeCalibration(
                        validIntrinsics(), new JsonMatOfDouble(2, 2, new double[] {0.1, 0.2, 0.3, 0.4}));

        assertMalformedCalibrationFallsBackToCpu(pipeline, cudaPipe, malformedIntrinsics);
        assertMalformedCalibrationFallsBackToCpu(pipeline, cudaPipe, malformedDistortion);

        assertEquals(
                List.of(CUDA_IDENTITY_CALIBRATION, CUDA_IDENTITY_CALIBRATION), cudaPipe.calibrationUpdates);
        assertEquals(2, cpuPipe.runCount);
        malformedIntrinsics.release();
        malformedDistortion.release();
        pipeline.release();
    }

    @Test
    public void extendedOpenCvCalibrationFallsBackToCpuDetection() {
        var cudaPipe = new FakeAprilTagDetectionCudaPipe();
        var cpuPipe = new FakeAprilTagDetectionPipe(List.of(makeDetection(14)));
        var pipeline = newCudaTestPipeline(cudaPipe, cpuPipe, new FakeAprilTagMLHybridPipe());
        pipeline.getSettings().useCudaTagDetection = true;
        pipeline.getSettings().useMLDetection = false;
        pipeline.getSettings().solvePNPEnabled = false;
        var extendedDistortion =
                makeCalibration(
                        validIntrinsics(),
                        new JsonMatOfDouble(
                                1,
                                8,
                                new double[] {0.1, -0.2, 0.003, -0.004, 0.05, 0.06, -0.07, 0.08}));
        var frame = makeFrame(extendedDistortion);

        var result = pipeline.run(frame, QuirkyCamera.DefaultCamera);

        assertEquals(List.of(false), cudaPipe.enabledTransitions);
        assertTrue(cudaPipe.calibrationUpdates.isEmpty());
        assertEquals(1, cpuPipe.runCount);
        assertEquals(1, result.targets.size());
        result.release();
        frame.release();
        extendedDistortion.release();
        pipeline.release();
    }

    @Test
    public void initiallyUnavailableCudaFallsBackToMl() {
        var cudaPipe = new FakeAprilTagDetectionCudaPipe();
        cudaPipe.available = false;
        var cpuPipe = new FakeAprilTagDetectionPipe(List.of(makeDetection(12)));
        var mlPipe = new FakeAprilTagMLHybridPipe();
        mlPipe.nextResult = new MLDetectionResult(List.of(makeDetection(13)), List.of(), 1);
        var pipeline = newCudaTestPipeline(cudaPipe, cpuPipe, mlPipe, Optional.of(new FakeModel()));
        pipeline.getSettings().useCudaTagDetection = true;
        pipeline.getSettings().useMLDetection = true;
        pipeline.getSettings().solvePNPEnabled = false;

        var frame = makeFrame();
        var result = pipeline.run(frame, QuirkyCamera.DefaultCamera);

        assertEquals(0, cudaPipe.runCount);
        assertEquals(1, mlPipe.runCount);
        assertEquals(0, cpuPipe.runCount);
        assertEquals(13, result.targets.get(0).getFiducialId());
        result.release();
        frame.release();
        pipeline.release();
    }

    @Test
    public void initiallyUnavailableCudaFallsBackToCpuWhenMlIsDisabled() {
        var cudaPipe = new FakeAprilTagDetectionCudaPipe();
        cudaPipe.available = false;
        var cpuPipe = new FakeAprilTagDetectionPipe(List.of(makeDetection(14)));
        var pipeline = newCudaTestPipeline(cudaPipe, cpuPipe, new FakeAprilTagMLHybridPipe());
        pipeline.getSettings().useCudaTagDetection = true;
        pipeline.getSettings().useMLDetection = false;
        pipeline.getSettings().solvePNPEnabled = false;

        var frame = makeFrame();
        var result = pipeline.run(frame, QuirkyCamera.DefaultCamera);

        assertEquals(0, cudaPipe.runCount);
        assertEquals(1, cpuPipe.runCount);
        assertEquals(14, result.targets.get(0).getFiducialId());
        result.release();
        frame.release();
        pipeline.release();
    }

    @Test
    public void initialMalformedCudaCalibrationSkipsAllPoseProcessingWithoutMutation() {
        var cudaPipe = new FakeAprilTagDetectionCudaPipe();
        cudaPipe.available = false;
        var posePipe = new FakeAprilTagPoseEstimatorPipe();
        var multiTagPipe = new FakeMultiTargetPNPPipe();
        var pipeline =
                newPoseTestPipeline(
                        cudaPipe,
                        new FakeAprilTagDetectionPipe(List.of(makeDetection(15))),
                        posePipe,
                        multiTagPipe);
        pipeline.getSettings().useCudaTagDetection = true;
        pipeline.getSettings().useMLDetection = false;
        pipeline.getSettings().solvePNPEnabled = true;
        pipeline.getSettings().doMultiTarget = true;
        var malformedCalibration =
                makeCalibration(new JsonMatOfDouble(1, 1, new double[] {700}), validDistortion());
        var frameProperties = framePropertiesWithUncheckedCalibration(malformedCalibration);
        var frame = makeFrame(frameProperties);

        var result = assertDoesNotThrow(() -> pipeline.run(frame, QuirkyCamera.DefaultCamera));

        assertSame(malformedCalibration, frameProperties.cameraCalibration);
        assertEquals(CUDA_IDENTITY_CALIBRATION, cudaPipe.lastCalibration);
        assertEquals(0, posePipe.setParamsCount);
        assertEquals(0, posePipe.runCount);
        assertEquals(0, multiTagPipe.setParamsCount);
        assertEquals(0, multiTagPipe.runCount);
        assertEquals(1, result.targets.size());
        result.release();
        frame.release();
        malformedCalibration.release();
        pipeline.release();
    }

    @Test
    public void validToMalformedCudaCalibrationDoesNotUseStalePoseParams() {
        var cudaPipe = new FakeAprilTagDetectionCudaPipe();
        cudaPipe.available = false;
        var posePipe = new FakeAprilTagPoseEstimatorPipe();
        var multiTagPipe = new FakeMultiTargetPNPPipe();
        var pipeline =
                newPoseTestPipeline(
                        cudaPipe,
                        new FakeAprilTagDetectionPipe(List.of(makeDetection(16))),
                        posePipe,
                        multiTagPipe);
        pipeline.getSettings().useCudaTagDetection = true;
        pipeline.getSettings().useMLDetection = false;
        pipeline.getSettings().solvePNPEnabled = true;
        pipeline.getSettings().doMultiTarget = true;
        var validCalibration = makeCalibration(validIntrinsics(), validDistortion());
        var malformedCalibration =
                makeCalibration(new JsonMatOfDouble(1, 1, new double[] {700}), validDistortion());

        var validFrame = makeFrame(validCalibration);
        var validResult = pipeline.run(validFrame, QuirkyCamera.DefaultCamera);
        validResult.release();
        validFrame.release();

        var malformedProperties = framePropertiesWithUncheckedCalibration(malformedCalibration);
        var malformedFrame = makeFrame(malformedProperties);
        var malformedResult =
                assertDoesNotThrow(() -> pipeline.run(malformedFrame, QuirkyCamera.DefaultCamera));

        assertSame(malformedCalibration, malformedProperties.cameraCalibration);
        assertEquals(1, posePipe.setParamsCount);
        assertEquals(1, posePipe.runCount);
        assertEquals(1, multiTagPipe.setParamsCount);
        assertEquals(1, multiTagPipe.runCount);
        malformedResult.release();
        malformedFrame.release();
        validCalibration.release();
        malformedCalibration.release();
        pipeline.release();
    }

    @Test
    public void malformedToValidCudaCalibrationRefreshesAndRestoresPoseProcessing() {
        var cudaPipe = new FakeAprilTagDetectionCudaPipe();
        cudaPipe.available = false;
        var posePipe = new FakeAprilTagPoseEstimatorPipe();
        var multiTagPipe = new FakeMultiTargetPNPPipe();
        var pipeline =
                newPoseTestPipeline(
                        cudaPipe,
                        new FakeAprilTagDetectionPipe(List.of(makeDetection(17))),
                        posePipe,
                        multiTagPipe);
        pipeline.getSettings().useCudaTagDetection = true;
        pipeline.getSettings().useMLDetection = false;
        pipeline.getSettings().solvePNPEnabled = true;
        pipeline.getSettings().doMultiTarget = true;
        var malformedCalibration =
                makeCalibration(new JsonMatOfDouble(1, 1, new double[] {700}), validDistortion());
        var validCalibration = makeCalibration(validIntrinsics(), validDistortion());
        var malformedProperties = framePropertiesWithUncheckedCalibration(malformedCalibration);

        var malformedFrame = makeFrame(malformedProperties);
        var malformedResult = pipeline.run(malformedFrame, QuirkyCamera.DefaultCamera);
        malformedResult.release();
        malformedFrame.release();

        var validFrame = makeFrame(validCalibration);
        var validResult = pipeline.run(validFrame, QuirkyCamera.DefaultCamera);

        assertSame(malformedCalibration, malformedProperties.cameraCalibration);
        assertEquals(1, posePipe.setParamsCount);
        assertEquals(1, posePipe.runCount);
        assertEquals(1, multiTagPipe.setParamsCount);
        assertEquals(1, multiTagPipe.runCount);
        assertEquals(
                List.of(
                        CUDA_IDENTITY_CALIBRATION,
                        new Calibration(700, 710, 320, 240, 0.1, -0.2, 0.003, -0.004, 0.05)),
                cudaPipe.calibrationUpdates);
        validResult.release();
        validFrame.release();
        malformedCalibration.release();
        validCalibration.release();
        pipeline.release();
    }

    @Test
    public void disablingCudaAfterInvalidCalibrationRestoresCpuOnlyPoseBehavior() {
        var cudaPipe = new FakeAprilTagDetectionCudaPipe();
        cudaPipe.available = false;
        var posePipe = new FakeAprilTagPoseEstimatorPipe();
        var multiTagPipe = new FakeMultiTargetPNPPipe();
        var pipeline =
                newPoseTestPipeline(
                        cudaPipe, new FakeAprilTagDetectionPipe(List.of()), posePipe, multiTagPipe);
        pipeline.getSettings().useCudaTagDetection = true;
        pipeline.getSettings().useMLDetection = false;
        pipeline.getSettings().solvePNPEnabled = true;
        pipeline.getSettings().doMultiTarget = true;
        var cpuAcceptedCalibration =
                makeCalibration(
                        validIntrinsics(), new JsonMatOfDouble(2, 2, new double[] {0.1, 0.2, 0.3, 0.4}));

        var cudaFrameProperties = framePropertiesWithUncheckedCalibration(cpuAcceptedCalibration);
        var cudaFrame = makeFrame(cudaFrameProperties);
        var cudaResult = pipeline.run(cudaFrame, QuirkyCamera.DefaultCamera);
        cudaResult.release();
        cudaFrame.release();

        pipeline.getSettings().useCudaTagDetection = false;
        var cpuFrameProperties = framePropertiesWithUncheckedCalibration(cpuAcceptedCalibration);
        var cpuFrame = makeFrame(cpuFrameProperties);
        var cpuResult = pipeline.run(cpuFrame, QuirkyCamera.DefaultCamera);

        assertSame(cpuAcceptedCalibration, cudaFrameProperties.cameraCalibration);
        assertSame(cpuAcceptedCalibration, cpuFrameProperties.cameraCalibration);
        assertEquals(1, posePipe.setParamsCount);
        assertEquals(0, posePipe.runCount);
        assertEquals(1, multiTagPipe.setParamsCount);
        assertEquals(1, multiTagPipe.runCount);
        cpuResult.release();
        cpuFrame.release();
        cpuAcceptedCalibration.release();
        pipeline.release();
    }

    private static void assertMalformedCalibrationFallsBackToCpu(
            TestAprilTagPipeline pipeline,
            FakeAprilTagDetectionCudaPipe cudaPipe,
            CameraCalibrationCoefficients calibration) {
        var frameStaticProperties = new FrameStaticProperties(640, 480, 70, null);
        frameStaticProperties.cameraCalibration = calibration;
        var frame = makeFrame(frameStaticProperties);

        var result = assertDoesNotThrow(() -> pipeline.run(frame, QuirkyCamera.DefaultCamera));

        assertSame(calibration, frameStaticProperties.cameraCalibration);
        assertEquals(CUDA_IDENTITY_CALIBRATION, cudaPipe.lastCalibration);
        assertEquals(1, result.targets.size());
        result.release();
        frame.release();
    }

    @Test
    public void disablingCudaOccursBeforeTheNextFrameUsesCpuDetection() {
        var events = new ArrayList<String>();
        var cudaPipe = new FakeAprilTagDetectionCudaPipe(events);
        var cpuPipe = new FakeAprilTagDetectionPipe(List.of(), events);
        var pipeline = newCudaTestPipeline(cudaPipe, cpuPipe, new FakeAprilTagMLHybridPipe());
        pipeline.getSettings().useCudaTagDetection = true;
        pipeline.getSettings().solvePNPEnabled = false;

        var firstFrame = makeFrame();
        var firstResult = pipeline.run(firstFrame, QuirkyCamera.DefaultCamera);
        firstResult.release();
        firstFrame.release();

        pipeline.getSettings().useCudaTagDetection = false;
        var secondFrame = makeFrame();
        var secondResult = pipeline.run(secondFrame, QuirkyCamera.DefaultCamera);

        assertTrue(events.indexOf("cuda-enabled-false") < events.lastIndexOf("cpu-run"));
        secondResult.release();
        secondFrame.release();
        pipeline.release();
    }

    @Test
    public void cudaFailureFallsBackToMlWhenMlIsEnabled() {
        var cudaPipe = new FakeAprilTagDetectionCudaPipe();
        cudaPipe.failDuringRun = true;
        var cpuPipe = new FakeAprilTagDetectionPipe(List.of(makeDetection(6)));
        var mlPipe = new FakeAprilTagMLHybridPipe();
        mlPipe.nextResult = new MLDetectionResult(List.of(makeDetection(7)), List.of(), 1);
        var pipeline = newCudaTestPipeline(cudaPipe, cpuPipe, mlPipe, Optional.of(new FakeModel()));
        pipeline.getSettings().useCudaTagDetection = true;
        pipeline.getSettings().useMLDetection = true;
        pipeline.getSettings().solvePNPEnabled = false;

        var frame = makeFrame();
        var result = pipeline.run(frame, QuirkyCamera.DefaultCamera);

        assertEquals(1, cudaPipe.runCount);
        assertEquals(1, mlPipe.runCount);
        assertEquals(0, cpuPipe.runCount);
        assertEquals(7, result.targets.get(0).getFiducialId());
        result.release();
        frame.release();
        pipeline.release();
    }

    @Test
    public void cudaFailureFallsBackToCpuWhenMlIsDisabled() {
        var cudaPipe = new FakeAprilTagDetectionCudaPipe();
        cudaPipe.failDuringRun = true;
        var cpuPipe = new FakeAprilTagDetectionPipe(List.of(makeDetection(8)));
        var pipeline = newCudaTestPipeline(cudaPipe, cpuPipe, new FakeAprilTagMLHybridPipe());
        pipeline.getSettings().useCudaTagDetection = true;
        pipeline.getSettings().useMLDetection = false;
        pipeline.getSettings().solvePNPEnabled = false;

        var frame = makeFrame();
        var result = pipeline.run(frame, QuirkyCamera.DefaultCamera);

        assertEquals(1, cudaPipe.runCount);
        assertEquals(1, cpuPipe.runCount);
        assertEquals(8, result.targets.get(0).getFiducialId());
        result.release();
        frame.release();
        pipeline.release();
    }

    @Test
    public void validEmptyCudaResultDoesNotInvokeMlOrCpuFallback() {
        var cudaPipe = new FakeAprilTagDetectionCudaPipe();
        var cpuPipe = new FakeAprilTagDetectionPipe(List.of(makeDetection(9)));
        var mlPipe = new FakeAprilTagMLHybridPipe();
        mlPipe.nextResult = new MLDetectionResult(List.of(makeDetection(10)), List.of(), 1);
        var pipeline = newCudaTestPipeline(cudaPipe, cpuPipe, mlPipe, Optional.of(new FakeModel()));
        pipeline.getSettings().useCudaTagDetection = true;
        pipeline.getSettings().useMLDetection = true;
        pipeline.getSettings().solvePNPEnabled = false;

        var frame = makeFrame();
        var result = pipeline.run(frame, QuirkyCamera.DefaultCamera);

        assertEquals(1, cudaPipe.runCount);
        assertEquals(0, mlPipe.runCount);
        assertEquals(0, cpuPipe.runCount);
        assertTrue(result.targets.isEmpty());
        result.release();
        frame.release();
        pipeline.release();
    }

    @Test
    public void pipelineReleaseReachesCudaPipe() {
        var cudaPipe = new FakeAprilTagDetectionCudaPipe();
        var pipeline =
                newCudaTestPipeline(
                        cudaPipe, new FakeAprilTagDetectionPipe(List.of()), new FakeAprilTagMLHybridPipe());

        pipeline.release();

        assertEquals(1, cudaPipe.releaseCount);
    }

    private static TestAprilTagPipeline newCudaTestPipeline(
            FakeAprilTagDetectionCudaPipe cudaPipe,
            AprilTagDetectionPipe aprilTagDetectionPipe,
            AprilTagMLHybridPipe mlHybridPipe,
            Optional<Model>... mlModels) {
        return new TestAprilTagPipeline(
                cudaPipe,
                aprilTagDetectionPipe,
                mlHybridPipe,
                new AprilTagPoseEstimatorPipe(),
                new MultiTargetPNPPipe(),
                new CalculateFPSPipe(),
                mlModels);
    }

    private static TestAprilTagPipeline newPoseTestPipeline(
            FakeAprilTagDetectionCudaPipe cudaPipe,
            AprilTagDetectionPipe aprilTagDetectionPipe,
            FakeAprilTagPoseEstimatorPipe posePipe,
            FakeMultiTargetPNPPipe multiTagPipe) {
        return new TestAprilTagPipeline(
                cudaPipe,
                aprilTagDetectionPipe,
                new FakeAprilTagMLHybridPipe(),
                posePipe,
                multiTagPipe,
                new CalculateFPSPipe());
    }

    private static AprilTagDetection makeDetection(int id) {
        return new AprilTagDetection(
                "tag36h11",
                id,
                0,
                100,
                new double[] {1, 0, 0, 0, 1, 0, 0, 0, 1},
                120,
                140,
                new double[] {100, 100, 140, 100, 140, 140, 100, 140});
    }

    private static Frame makeFrame() {
        return makeFrame(new FrameStaticProperties(640, 480, 70, null));
    }

    private static Frame makeFrame(CameraCalibrationCoefficients calibration) {
        return makeFrame(new FrameStaticProperties(640, 480, 70, calibration));
    }

    private static Frame makeFrame(FrameStaticProperties frameStaticProperties) {
        return new Frame(
                1,
                new CVMat(Mat.zeros(480, 640, CvType.CV_8UC3)),
                new CVMat(Mat.zeros(480, 640, CvType.CV_8UC1)),
                FrameThresholdType.GREYSCALE,
                frameStaticProperties);
    }

    private static FrameStaticProperties framePropertiesWithUncheckedCalibration(
            CameraCalibrationCoefficients calibration) {
        var frameStaticProperties = new FrameStaticProperties(640, 480, 70, null);
        frameStaticProperties.cameraCalibration = calibration;
        return frameStaticProperties;
    }

    private static CameraCalibrationCoefficients makeCalibration(
            JsonMatOfDouble intrinsics, JsonMatOfDouble distortion) {
        return new CameraCalibrationCoefficients(
                new Size(640, 480),
                intrinsics,
                distortion,
                new double[0],
                List.of(),
                new Size(),
                0,
                CameraLensModel.LENSMODEL_OPENCV);
    }

    private static JsonMatOfDouble validIntrinsics() {
        return new JsonMatOfDouble(3, 3, new double[] {700, 0, 320, 0, 710, 240, 0, 0, 1});
    }

    private static JsonMatOfDouble validDistortion() {
        return new JsonMatOfDouble(1, 5, new double[] {0.1, -0.2, 0.003, -0.004, 0.05});
    }

    private static final class FakeAprilTagDetectionPipe extends AprilTagDetectionPipe {
        private final List<AprilTagDetection> detections;
        private final List<String> events;
        private int runCount;

        private FakeAprilTagDetectionPipe(List<AprilTagDetection> detections) {
            this(detections, null);
        }

        private FakeAprilTagDetectionPipe(List<AprilTagDetection> detections, List<String> events) {
            this.detections = detections;
            this.events = events;
        }

        @Override
        public CVPipeResult<List<AprilTagDetection>> run(CVMat in) {
            runCount++;
            if (events != null) {
                events.add("cpu-run");
            }
            var result = new CVPipeResult<List<AprilTagDetection>>();
            result.output = detections;
            result.nanosElapsed = 5;
            return result;
        }

        @Override
        public void release() {}
    }

    private static final class FakeAprilTagDetectionCudaPipe extends AprilTagDetectionCudaPipe {
        private final List<Boolean> enabledTransitions = new ArrayList<>();
        private final List<String> events;
        private boolean enabled;
        private boolean available = true;
        private boolean failDuringRun;
        private int runCount;
        private int releaseCount;
        private AprilTagDetectionCudaPipeParams lastParams;
        private Calibration lastCalibration;
        private final List<Calibration> calibrationUpdates = new ArrayList<>();

        private FakeAprilTagDetectionCudaPipe() {
            this(new ArrayList<>());
        }

        private FakeAprilTagDetectionCudaPipe(List<String> events) {
            super(new InertCudaBackend());
            this.events = events;
        }

        @Override
        public void setEnabled(boolean enabled) {
            this.enabled = enabled;
            enabledTransitions.add(enabled);
            events.add("cuda-enabled-" + enabled);
        }

        @Override
        public boolean isAvailable() {
            return enabled && available;
        }

        @Override
        public void setParams(AprilTagDetectionCudaPipeParams newParams) {
            lastParams = newParams;
        }

        @Override
        public void setCalibration(Calibration calibration) {
            lastCalibration = calibration;
            calibrationUpdates.add(calibration);
        }

        @Override
        public CVPipeResult<List<AprilTagDetection>> run(CVMat in) {
            runCount++;
            events.add("cuda-run");
            if (failDuringRun) {
                available = false;
            }
            var result = new CVPipeResult<List<AprilTagDetection>>();
            result.output = List.of();
            return result;
        }

        @Override
        public void release() {
            releaseCount++;
        }
    }

    private static final class FakeAprilTagPoseEstimatorPipe extends AprilTagPoseEstimatorPipe {
        private int setParamsCount;
        private int runCount;

        @Override
        public void setParams(AprilTagPoseEstimatorPipeParams newParams) {
            setParamsCount++;
        }

        @Override
        public CVPipeResult<AprilTagPoseEstimate> run(AprilTagDetection detection) {
            runCount++;
            var result = new CVPipeResult<AprilTagPoseEstimate>();
            result.output = null;
            return result;
        }

        @Override
        public void release() {}
    }

    private static final class FakeMultiTargetPNPPipe extends MultiTargetPNPPipe {
        private int setParamsCount;
        private int runCount;

        @Override
        public void setParams(MultiTargetPNPPipeParams newParams) {
            setParamsCount++;
        }

        @Override
        public CVPipeResult<Optional<MultiTargetPNPResult>> run(List<TrackedTarget> targets) {
            runCount++;
            var result = new CVPipeResult<Optional<MultiTargetPNPResult>>();
            result.output = Optional.empty();
            return result;
        }
    }

    private static final class InertCudaBackend implements AprilTagDetectionCudaPipe.Backend {
        @Override
        public boolean isAvailable() {
            return false;
        }

        @Override
        public String getLoadError() {
            return "not used by pipeline fake";
        }

        @Override
        public long createGpuDetector(int width, int height, int decimate) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void setCalibration(
                long handle,
                double fx,
                double fy,
                double cx,
                double cy,
                double k1,
                double k2,
                double p1,
                double p2,
                double k3) {
            throw new UnsupportedOperationException();
        }

        @Override
        public AprilTagDetection[] processGray(
                long handle, long dataAddress, int width, int height, long strideBytes) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void destroyGpuDetector(long handle) {
            throw new UnsupportedOperationException();
        }

        @Override
        public long getCudaFreeMemoryBytes() {
            return 0;
        }

        @Override
        public long getCudaTotalMemoryBytes() {
            return 0;
        }

        @Override
        public String getBuildInfo() {
            return "inert";
        }
    }

    private static final class FakeAprilTagMLHybridPipe extends AprilTagMLHybridPipe {
        private MLDetectionResult nextResult = new MLDetectionResult(List.of(), List.of(), 0);
        private Model lastModel;
        private int runCount;

        @Override
        public void setParams(Params newParams) {
            lastModel = newParams.detectionParams.model;
        }

        @Override
        public CVPipeResult<MLDetectionResult> run(Frame frame) {
            runCount++;
            var result = new CVPipeResult<MLDetectionResult>();
            result.output = nextResult;
            result.nanosElapsed = nextResult.nanosElapsed();
            return result;
        }

        @Override
        public boolean isAvailable() {
            return lastModel != null && !(lastModel instanceof NullModel);
        }

        @Override
        public void release() {}
    }

    private static final class TestAprilTagPipeline extends AprilTagPipeline {
        private final Queue<Optional<Model>> mlModels;

        private TestAprilTagPipeline(
                AprilTagDetectionPipe aprilTagDetectionPipe,
                AprilTagMLHybridPipe mlHybridPipe,
                AprilTagPoseEstimatorPipe singleTagPoseEstimatorPipe,
                MultiTargetPNPPipe multiTagPNPPipe,
                CalculateFPSPipe calculateFPSPipe,
                Optional<Model>... mlModels) {
            this(
                    new FakeAprilTagDetectionCudaPipe(),
                    aprilTagDetectionPipe,
                    mlHybridPipe,
                    singleTagPoseEstimatorPipe,
                    multiTagPNPPipe,
                    calculateFPSPipe,
                    mlModels);
        }

        private TestAprilTagPipeline(
                AprilTagDetectionCudaPipe cudaDetectionPipe,
                AprilTagDetectionPipe aprilTagDetectionPipe,
                AprilTagMLHybridPipe mlHybridPipe,
                AprilTagPoseEstimatorPipe singleTagPoseEstimatorPipe,
                MultiTargetPNPPipe multiTagPNPPipe,
                CalculateFPSPipe calculateFPSPipe,
                Optional<Model>... mlModels) {
            super(
                    new AprilTagPipelineSettings(),
                    aprilTagDetectionPipe,
                    cudaDetectionPipe,
                    mlHybridPipe,
                    singleTagPoseEstimatorPipe,
                    multiTagPNPPipe,
                    calculateFPSPipe);
            this.mlModels = new ArrayDeque<>(List.of(mlModels));
        }

        @Override
        Optional<Model> getMlModel() {
            return mlModels.isEmpty() ? Optional.empty() : mlModels.remove();
        }
    }

    private static final class FakeModel implements Model {
        @Override
        public org.photonvision.vision.objects.ObjectDetector load() {
            return NullModel.getInstance();
        }

        @Override
        public String getUID() {
            return "fake-ml-model";
        }

        @Override
        public String getNickname() {
            return "fake-ml-model";
        }

        @Override
        public Family getFamily() {
            return Family.TENSORRT;
        }

        @Override
        public ModelProperties getProperties() {
            return new ModelProperties(
                    Path.of("fake-ml-model.onnx"),
                    "fake-ml-model",
                    List.of("AprilTag"),
                    640,
                    640,
                    Family.TENSORRT,
                    Version.YOLOV8);
        }

        @Override
        public String toString() {
            return getUID();
        }
    }

    private static final class ThrowingModel implements Model {
        @Override
        public org.photonvision.vision.objects.ObjectDetector load() {
            throw new RuntimeException("boom");
        }

        @Override
        public String getUID() {
            return "throwing-ml-model";
        }

        @Override
        public String getNickname() {
            return "throwing-ml-model";
        }

        @Override
        public Family getFamily() {
            return Family.TENSORRT;
        }

        @Override
        public ModelProperties getProperties() {
            return new ModelProperties(
                    Path.of("throwing-ml-model.onnx"),
                    "throwing-ml-model",
                    List.of("AprilTag"),
                    640,
                    640,
                    Family.TENSORRT,
                    Version.YOLOV8);
        }
    }
}
