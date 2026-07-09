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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import edu.wpi.first.apriltag.AprilTagDetection;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.List;
import java.util.Optional;
import java.util.Queue;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.RotatedRect;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Family;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Version;
import org.photonvision.common.configuration.NeuralNetworkModelsSettings.ModelProperties;
import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.objects.Model;
import org.photonvision.vision.objects.NullModel;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipe.CVPipe.CVPipeResult;
import org.photonvision.vision.pipe.impl.AprilTagDetectionPipe;
import org.photonvision.vision.pipe.impl.AprilTagMLHybridPipe;
import org.photonvision.vision.pipe.impl.AprilTagPoseEstimatorPipe;
import org.photonvision.vision.pipe.impl.CalculateFPSPipe;
import org.photonvision.vision.pipe.impl.MLDetectionResult;
import org.photonvision.vision.pipe.impl.MultiTargetPNPPipe;

public class AprilTagPipelineMLBehaviorTest {
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
        return new Frame(
                1,
                new CVMat(Mat.zeros(480, 640, CvType.CV_8UC3)),
                new CVMat(Mat.zeros(480, 640, CvType.CV_8UC1)),
                FrameThresholdType.GREYSCALE,
                new FrameStaticProperties(640, 480, 70, null));
    }

    private static final class FakeAprilTagDetectionPipe extends AprilTagDetectionPipe {
        private final List<AprilTagDetection> detections;
        private int runCount;

        private FakeAprilTagDetectionPipe(List<AprilTagDetection> detections) {
            this.detections = detections;
        }

        @Override
        public CVPipeResult<List<AprilTagDetection>> run(CVMat in) {
            runCount++;
            var result = new CVPipeResult<List<AprilTagDetection>>();
            result.output = detections;
            result.nanosElapsed = 5;
            return result;
        }

        @Override
        public void release() {}
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
            super(
                    new AprilTagPipelineSettings(),
                    aprilTagDetectionPipe,
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
