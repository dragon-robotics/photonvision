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

package org.photonvision.vision.pipe;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Family;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Version;
import org.photonvision.common.configuration.NeuralNetworkModelsSettings.ModelProperties;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.objects.Model;
import org.photonvision.vision.objects.ObjectDetector;
import org.photonvision.vision.pipe.impl.AprilTagMLHybridPipe;
import org.photonvision.vision.pipe.impl.AprilTagROIDetectionPipe;
import org.photonvision.vision.pipe.impl.NeuralNetworkPipeResult;

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

    @Test
    public void roiDetectionPipeClearsDetectorWhenModelBecomesNull() {
        var detector = new FakeObjectDetector();
        var model = new FakeModel(detector);
        var pipe = new AprilTagROIDetectionPipe();

        pipe.setParams(new AprilTagROIDetectionPipe.AprilTagROIDetectionParams(model, 0.5, 0.45));
        assertTrue(pipe.isAvailable());

        pipe.setParams(new AprilTagROIDetectionPipe.AprilTagROIDetectionParams(null, 0.5, 0.45));
        assertFalse(pipe.isAvailable());
        assertTrue(detector.released);

        pipe.release();
    }

    private static final class FakeModel implements Model {
        private final FakeObjectDetector detector;

        private FakeModel(FakeObjectDetector detector) {
            this.detector = detector;
        }

        @Override
        public ObjectDetector load() {
            return detector;
        }

        @Override
        public String getUID() {
            return "fake-model";
        }

        @Override
        public String getNickname() {
            return "fake-model";
        }

        @Override
        public Family getFamily() {
            return Family.TENSORRT;
        }

        @Override
        public ModelProperties getProperties() {
            return new ModelProperties(
                    java.nio.file.Path.of("fake.onnx"),
                    "fake-model",
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

    private static final class FakeObjectDetector implements ObjectDetector {
        private boolean released;

        @Override
        public Model getModel() {
            return null;
        }

        @Override
        public List<String> getClasses() {
            return List.of("AprilTag");
        }

        @Override
        public List<NeuralNetworkPipeResult> detect(Mat in, double nmsThresh, double boxThresh) {
            return List.of();
        }

        @Override
        public void release() {
            released = true;
        }
    }
}
