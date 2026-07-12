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

package org.photonvision.vision.pipe.impl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import edu.wpi.first.apriltag.AprilTagDetection;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.photonvision.jni.LibraryLoader;
import org.photonvision.vision.opencv.CVMat;

class AprilTagDetectionCudaPipeTest {
    @BeforeAll
    static void loadLibraries() {
        assertTrue(LibraryLoader.loadWpiLibraries());
    }

    @Test
    void disabledPipeNeverCreatesDetector() {
        var backend = new FakeBackend();
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var image = grayImage(640, 480);
        try {
            pipe.setEnabled(false);

            assertTrue(pipe.run(image).output.isEmpty());
            assertEquals(0, backend.createCount);
        } finally {
            image.release();
            pipe.release();
        }
    }

    @Test
    void unavailableBackendNeverCreatesDetector() {
        var backend = new FakeBackend();
        backend.available = false;
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var image = grayImage(640, 480);
        try {
            pipe.setEnabled(true);

            assertTrue(pipe.run(image).output.isEmpty());
            assertEquals(0, backend.createCount);
            assertFalse(pipe.isAvailable());
        } finally {
            image.release();
            pipe.release();
        }
    }

    @Test
    void disablingDestroysHandleAndReenableCreatesFreshHandle() {
        var backend = new FakeBackend();
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var image = grayImage(640, 480);
        try {
            pipe.setEnabled(true);
            pipe.run(image);
            pipe.setEnabled(false);

            assertEquals(1, backend.destroyCount);
            pipe.setEnabled(true);
            pipe.run(image);
            assertEquals(2, backend.createCount);
        } finally {
            image.release();
            pipe.release();
        }
    }

    @Test
    void nativeDecimateTwoReceivesFullResolutionAndRawFrameMetadata() {
        var backend = new FakeBackend();
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var image = grayImage(1600, 1304);
        try {
            pipe.setEnabled(true);
            pipe.setParams(new AprilTagDetectionCudaPipe.AprilTagDetectionCudaPipeParams(2));
            pipe.run(image);

            assertEquals(1600, backend.createdWidth);
            assertEquals(1304, backend.createdHeight);
            assertEquals(2, backend.createdNativeDecimate);
            assertEquals(image.getMat().dataAddr(), backend.detectedDataAddress);
            assertEquals(1600, backend.detectedWidth);
            assertEquals(1304, backend.detectedHeight);
            assertEquals(image.getMat().step1(), backend.detectedStrideBytes);
            assertEquals(1, backend.processCount);
        } finally {
            image.release();
            pipe.release();
        }
    }

    @Test
    void javaDecimateThreeResizesAndScalesEachOutputCoordinateOnce() {
        var backend = new FakeBackend();
        backend.detections = new AprilTagDetection[] {detection(10, 20)};
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var image = grayImage(1600, 1304);
        try {
            pipe.setEnabled(true);
            pipe.setParams(new AprilTagDetectionCudaPipe.AprilTagDetectionCudaPipeParams(3));

            var output = pipe.run(image).output;
            double xScale = 1600.0 / 528.0;
            double yScale = 1304.0 / 432.0;

            assertEquals(528, backend.createdWidth);
            assertEquals(432, backend.createdHeight);
            assertEquals(1, backend.createdNativeDecimate);
            assertEquals(10 * xScale, output.get(0).getCenterX(), 1e-9);
            assertEquals(20 * yScale, output.get(0).getCenterY(), 1e-9);
            assertEquals(8 * xScale, output.get(0).getCornerX(0), 1e-9);
            assertEquals(25 * yScale, output.get(0).getCornerY(3), 1e-9);
            assertEquals(1 * xScale, output.get(0).getHomography()[0], 1e-9);
            assertEquals(6 * yScale, output.get(0).getHomography()[5], 1e-9);
            assertEquals(7, output.get(0).getHomography()[6], 1e-9);
        } finally {
            image.release();
            pipe.release();
        }
    }

    @Test
    void invalidDecimationClampsToOne() {
        var backend = new FakeBackend();
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var image = grayImage(640, 480);
        try {
            pipe.setEnabled(true);
            pipe.setParams(new AprilTagDetectionCudaPipe.AprilTagDetectionCudaPipeParams(0));
            pipe.run(image);

            assertEquals(640, backend.createdWidth);
            assertEquals(480, backend.createdHeight);
            assertEquals(1, backend.createdNativeDecimate);
        } finally {
            image.release();
            pipe.release();
        }
    }

    @Test
    void createFailureCanRecoverAfterDisableAndRetry() {
        var backend = new FakeBackend();
        backend.failCreate = true;
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var image = grayImage(640, 480);
        try {
            pipe.setEnabled(true);

            assertTrue(pipe.run(image).output.isEmpty());
            assertEquals(1, backend.createCount);
            assertEquals(0, backend.destroyCount);
            assertFalse(pipe.isAvailable());

            pipe.setEnabled(false);
            backend.failCreate = false;
            pipe.setEnabled(true);
            pipe.run(image);

            assertEquals(2, backend.createCount);
            assertEquals(1, backend.processCount);
            assertTrue(pipe.isAvailable());
        } finally {
            image.release();
            pipe.release();
        }
    }

    @Test
    void calibrationFailureDestroysNewHandleAndCanRecoverAfterDisable() {
        var backend = new FakeBackend();
        backend.failCalibration = true;
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var image = grayImage(640, 480);
        try {
            pipe.setEnabled(true);

            assertTrue(pipe.run(image).output.isEmpty());
            assertEquals(1, backend.createCount);
            assertEquals(1, backend.calibrationCount);
            assertEquals(1, backend.destroyCount);
            assertEquals(0, backend.processCount);
            assertFalse(pipe.isAvailable());

            pipe.setEnabled(false);
            backend.failCalibration = false;
            pipe.setEnabled(true);
            pipe.run(image);

            assertEquals(2, backend.createCount);
            assertEquals(2, backend.calibrationCount);
            assertEquals(1, backend.processCount);
            assertTrue(pipe.isAvailable());
        } finally {
            image.release();
            pipe.release();
        }
    }

    @Test
    void runtimeFailureDestroysHandleBeforeFallbackState() {
        var backend = new FakeBackend();
        backend.failDetection = true;
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var image = grayImage(640, 480);
        try {
            pipe.setEnabled(true);

            assertTrue(pipe.run(image).output.isEmpty());
            assertEquals(List.of("create", "detect", "destroy"), backend.events);
            assertEquals(1, backend.destroyCount);
            assertFalse(pipe.isAvailable());
        } finally {
            image.release();
            pipe.release();
        }
    }

    @Test
    void successfulDisableRecoversFromRuntimeFailure() {
        var backend = new FakeBackend();
        backend.failDetection = true;
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var image = grayImage(640, 480);
        try {
            pipe.setEnabled(true);
            pipe.run(image);
            pipe.setEnabled(false);
            backend.failDetection = false;
            pipe.setEnabled(true);

            pipe.run(image);
            assertEquals(2, backend.createCount);
            assertTrue(pipe.isAvailable());
        } finally {
            image.release();
            pipe.release();
        }
    }

    @Test
    void destroyFailureCannotBeReenabledOrReconfigured() {
        var backend = new FakeBackend();
        backend.failDestroy = true;
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var image = grayImage(640, 480);
        try {
            pipe.setEnabled(true);
            pipe.run(image);
            pipe.setEnabled(false);
            pipe.setParams(new AprilTagDetectionCudaPipe.AprilTagDetectionCudaPipeParams(2));
            pipe.setEnabled(true);
            pipe.run(image);

            assertEquals(1, backend.createCount);
            assertEquals(1, backend.destroyCount);
            assertFalse(pipe.isAvailable());
        } finally {
            image.release();
            pipe.release();
        }
    }

    @Test
    void releaseAfterUncertainCleanupIsTerminalAndIdempotent() {
        var backend = new FakeBackend();
        backend.failDestroy = true;
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var image = grayImage(640, 480);
        try {
            pipe.setEnabled(true);
            pipe.run(image);
            var originalParams = pipe.getParams();
            pipe.setEnabled(false);

            pipe.release();
            pipe.release();
            pipe.setEnabled(true);
            pipe.setParams(new AprilTagDetectionCudaPipe.AprilTagDetectionCudaPipeParams(2));
            pipe.setCalibration(new AprilTagDetectionCudaPipe.Calibration(2, 2, 1, 1, 0, 0, 0, 0, 0));
            pipe.run(image);

            assertEquals(originalParams, pipe.getParams());
            assertEquals(1, backend.createCount);
            assertEquals(1, backend.calibrationCount);
            assertEquals(1, backend.processCount);
            assertEquals(1, backend.destroyCount);
            assertFalse(pipe.isAvailable());
        } finally {
            image.release();
            pipe.release();
        }
    }

    @Test
    void nullNativeOutputFailsAndDestroysTheDetector() {
        var backend = new FakeBackend();
        backend.returnNullOutput = true;
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var image = grayImage(640, 480);
        try {
            pipe.setEnabled(true);

            assertTrue(pipe.run(image).output.isEmpty());
            assertEquals(1, backend.destroyCount);
            assertFalse(pipe.isAvailable());
        } finally {
            image.release();
            pipe.release();
        }
    }

    @Test
    void emptyNativeOutputIsHealthyAndDoesNotFallBack() {
        var backend = new FakeBackend();
        backend.detections = new AprilTagDetection[0];
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var image = grayImage(640, 480);
        try {
            pipe.setEnabled(true);

            assertTrue(pipe.run(image).output.isEmpty());
            assertEquals(0, backend.destroyCount);
            assertTrue(pipe.isAvailable());
        } finally {
            image.release();
            pipe.release();
        }
    }

    @Test
    void dimensionsAndEffectiveNativeDecimationRecreateTheDetector() {
        var backend = new FakeBackend();
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var firstImage = grayImage(640, 480);
        var secondImage = grayImage(800, 480);
        try {
            pipe.setEnabled(true);
            pipe.run(firstImage);
            pipe.run(secondImage);
            pipe.setParams(new AprilTagDetectionCudaPipe.AprilTagDetectionCudaPipeParams(2));
            pipe.run(secondImage);

            assertEquals(3, backend.createCount);
            assertEquals(2, backend.destroyCount);
            assertEquals(2, backend.createdNativeDecimate);
        } finally {
            firstImage.release();
            secondImage.release();
            pipe.release();
        }
    }

    @Test
    void unalignedNativeInputResizesAndScalesCalibration() {
        var backend = new FakeBackend();
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var image = grayImage(1919, 1079);
        try {
            pipe.setEnabled(true);
            pipe.setCalibration(
                    new AprilTagDetectionCudaPipe.Calibration(1000, 800, 700, 600, 1, 2, 3, 4, 5));

            pipe.run(image);

            assertEquals(1912, backend.createdWidth);
            assertEquals(1072, backend.createdHeight);
            assertEquals(1, backend.createdNativeDecimate);
            assertEquals(1912, backend.detectedWidth);
            assertEquals(1072, backend.detectedHeight);
            assertEquals(1000 * 1912.0 / 1919.0, backend.fx, 1e-9);
            assertEquals(800 * 1072.0 / 1079.0, backend.fy, 1e-9);
            assertEquals(700 * 1912.0 / 1919.0, backend.cx, 1e-9);
            assertEquals(600 * 1072.0 / 1079.0, backend.cy, 1e-9);
        } finally {
            image.release();
            pipe.release();
        }
    }

    @Test
    void calibrationScalesOnlyForJavaResizeAndAvoidsRedundantUpdates() {
        var backend = new FakeBackend();
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var image = grayImage(1600, 1304);
        try {
            pipe.setEnabled(true);
            pipe.setParams(new AprilTagDetectionCudaPipe.AprilTagDetectionCudaPipeParams(3));
            pipe.setCalibration(
                    new AprilTagDetectionCudaPipe.Calibration(1000, 800, 700, 600, 1, 2, 3, 4, 5));
            pipe.run(image);
            pipe.run(image);

            assertEquals(1, backend.calibrationCount);
            assertEquals(1000 * 528.0 / 1600.0, backend.fx, 1e-9);
            assertEquals(800 * 432.0 / 1304.0, backend.fy, 1e-9);
            assertEquals(700 * 528.0 / 1600.0, backend.cx, 1e-9);
            assertEquals(600 * 432.0 / 1304.0, backend.cy, 1e-9);
            assertEquals(1, backend.k1, 1e-9);
            assertEquals(5, backend.k3, 1e-9);
        } finally {
            image.release();
            pipe.release();
        }
    }

    @Test
    void sameAlignedDetectorSizeRecreatesForNewSourceCalibrationScale() {
        var backend = new FakeBackend();
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var firstImage = grayImage(1600, 1304);
        var secondImage = grayImage(1599, 1303);
        try {
            pipe.setEnabled(true);
            pipe.setParams(new AprilTagDetectionCudaPipe.AprilTagDetectionCudaPipeParams(3));
            pipe.setCalibration(
                    new AprilTagDetectionCudaPipe.Calibration(1000, 800, 700, 600, 1, 2, 3, 4, 5));
            pipe.run(firstImage);
            pipe.run(secondImage);

            assertEquals(2, backend.createCount);
            assertEquals(1, backend.destroyCount);
            assertEquals(2, backend.calibrationCount);
            assertEquals(1000 * 528.0 / 1599.0, backend.fx, 1e-9);
            assertEquals(800 * 432.0 / 1303.0, backend.fy, 1e-9);
            assertEquals(700 * 528.0 / 1599.0, backend.cx, 1e-9);
            assertEquals(600 * 432.0 / 1303.0, backend.cy, 1e-9);
        } finally {
            firstImage.release();
            secondImage.release();
            pipe.release();
        }
    }

    @Test
    void nativeToJavaResizeRecreatesWhenDetectorDimensionsMatch() {
        var backend = new FakeBackend();
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var nativeImage = grayImage(528, 432);
        var resizedImage = grayImage(1599, 1303);
        try {
            pipe.setEnabled(true);
            pipe.setCalibration(
                    new AprilTagDetectionCudaPipe.Calibration(1000, 800, 700, 600, 1, 2, 3, 4, 5));
            pipe.run(nativeImage);
            pipe.setParams(new AprilTagDetectionCudaPipe.AprilTagDetectionCudaPipeParams(3));
            pipe.run(resizedImage);

            assertEquals(2, backend.createCount);
            assertEquals(1, backend.destroyCount);
            assertEquals(2, backend.calibrationCount);
            assertEquals(1000 * 528.0 / 1599.0, backend.fx, 1e-9);
            assertEquals(800 * 432.0 / 1303.0, backend.fy, 1e-9);
        } finally {
            nativeImage.release();
            resizedImage.release();
            pipe.release();
        }
    }

    @Test
    void nativeDecimationCalibrationIsNotScaled() {
        var backend = new FakeBackend();
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var image = grayImage(1600, 1304);
        try {
            pipe.setEnabled(true);
            pipe.setParams(new AprilTagDetectionCudaPipe.AprilTagDetectionCudaPipeParams(2));
            pipe.setCalibration(
                    new AprilTagDetectionCudaPipe.Calibration(1000, 800, 700, 600, 1, 2, 3, 4, 5));
            pipe.run(image);

            assertEquals(1000, backend.fx, 1e-9);
            assertEquals(800, backend.fy, 1e-9);
            assertEquals(700, backend.cx, 1e-9);
            assertEquals(600, backend.cy, 1e-9);
        } finally {
            image.release();
            pipe.release();
        }
    }

    @Test
    void nonGrayscaleAndEmptyInputFailThroughTheFallbackState() {
        var backend = new FakeBackend();
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var validImage = grayImage(640, 480);
        var colorImage = new CVMat(Mat.zeros(480, 640, CvType.CV_8UC3));
        var emptyImage = new CVMat();
        try {
            pipe.setEnabled(true);
            pipe.run(validImage);

            assertTrue(pipe.run(colorImage).output.isEmpty());
            assertEquals(1, backend.destroyCount);
            assertFalse(pipe.isAvailable());
            pipe.setEnabled(false);
            pipe.setEnabled(true);
            pipe.run(validImage);
            assertTrue(pipe.run(emptyImage).output.isEmpty());
            assertEquals(2, backend.destroyCount);
            assertFalse(pipe.isAvailable());
        } finally {
            validImage.release();
            colorImage.release();
            emptyImage.release();
            pipe.release();
        }
    }

    @Test
    void diagnosticsFailureDoesNotMaskDetectorLifecycle() {
        var backend = new FakeBackend();
        backend.failDiagnostics = true;
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var image = grayImage(640, 480);
        try {
            pipe.setEnabled(true);

            assertTrue(pipe.run(image).output.isEmpty());
            assertTrue(pipe.isAvailable());
            pipe.setEnabled(false);
            assertEquals(1, backend.destroyCount);
        } finally {
            image.release();
            pipe.release();
        }
    }

    @Test
    void releaseIsTerminalIdempotentAndDoesNotLeakResizeMats() {
        int originalMatCount = CVMat.getMatCount();
        var backend = new FakeBackend();
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var image = grayImage(1600, 1304);
        try {
            pipe.setEnabled(true);
            pipe.setParams(new AprilTagDetectionCudaPipe.AprilTagDetectionCudaPipeParams(3));
            pipe.run(image);
            pipe.release();
            pipe.release();

            assertEquals(1, backend.destroyCount);
        } finally {
            image.release();
        }
        assertEquals(originalMatCount, CVMat.getMatCount());
    }

    @Test
    void resizeFailureReleasesTemporaryCvMat() {
        int originalMatCount = CVMat.getMatCount();
        var backend = new FakeBackend();
        var pipe = new AprilTagDetectionCudaPipe(backend);
        var image = new CVMat(new ResizeFailingMat());
        try {
            int matCountWithInput = CVMat.getMatCount();
            pipe.setEnabled(true);
            pipe.setParams(new AprilTagDetectionCudaPipe.AprilTagDetectionCudaPipeParams(3));

            assertTrue(pipe.run(image).output.isEmpty());
            assertEquals(matCountWithInput, CVMat.getMatCount());
            assertEquals(0, backend.createCount);
            assertFalse(pipe.isAvailable());
        } finally {
            image.release();
            pipe.release();
        }
        assertEquals(originalMatCount, CVMat.getMatCount());
    }

    private static CVMat grayImage(int width, int height) {
        return new CVMat(Mat.zeros(height, width, CvType.CV_8UC1));
    }

    private static AprilTagDetection detection(double centerX, double centerY) {
        return new AprilTagDetection(
                "tag36h11",
                7,
                0,
                100,
                new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9},
                centerX,
                centerY,
                new double[] {8, 9, 10, 11, 12, 13, 14, 25});
    }

    private static final class ResizeFailingMat extends Mat {
        @Override
        public boolean empty() {
            return false;
        }

        @Override
        public int type() {
            return CvType.CV_8UC1;
        }

        @Override
        public int cols() {
            return 640;
        }

        @Override
        public int rows() {
            return 480;
        }
    }

    private static final class FakeBackend implements AprilTagDetectionCudaPipe.Backend {
        boolean available = true;
        boolean failCreate;
        boolean failCalibration;
        boolean failDetection;
        boolean failDestroy;
        boolean failDiagnostics;
        boolean returnNullOutput;
        int createCount;
        int destroyCount;
        int calibrationCount;
        int processCount;
        int createdWidth;
        int createdHeight;
        int createdNativeDecimate;
        long detectedDataAddress;
        int detectedWidth;
        int detectedHeight;
        long detectedStrideBytes;
        double fx;
        double fy;
        double cx;
        double cy;
        double k1;
        double k3;
        AprilTagDetection[] detections = new AprilTagDetection[0];
        List<String> events = new ArrayList<>();

        @Override
        public boolean isAvailable() {
            return available;
        }

        @Override
        public String getLoadError() {
            return available ? null : "missing test backend";
        }

        @Override
        public long createGpuDetector(int width, int height, int decimate) {
            events.add("create");
            createCount++;
            createdWidth = width;
            createdHeight = height;
            createdNativeDecimate = decimate;
            if (failCreate) {
                throw new RuntimeException("create failed");
            }
            return createCount;
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
            calibrationCount++;
            this.fx = fx;
            this.fy = fy;
            this.cx = cx;
            this.cy = cy;
            this.k1 = k1;
            this.k3 = k3;
            if (failCalibration) {
                throw new RuntimeException("calibration failed");
            }
        }

        @Override
        public AprilTagDetection[] processGray(
                long handle, long dataAddress, int width, int height, long strideBytes) {
            events.add("detect");
            processCount++;
            detectedDataAddress = dataAddress;
            detectedWidth = width;
            detectedHeight = height;
            detectedStrideBytes = strideBytes;
            if (failDetection) {
                throw new RuntimeException("detect failed");
            }
            return returnNullOutput ? null : detections;
        }

        @Override
        public void destroyGpuDetector(long handle) {
            events.add("destroy");
            destroyCount++;
            if (failDestroy) {
                throw new RuntimeException("destroy failed");
            }
        }

        @Override
        public long getCudaFreeMemoryBytes() {
            if (failDiagnostics) {
                throw new RuntimeException("diagnostics failed");
            }
            return 1_000;
        }

        @Override
        public long getCudaTotalMemoryBytes() {
            if (failDiagnostics) {
                throw new RuntimeException("diagnostics failed");
            }
            return 2_000;
        }

        @Override
        public String getBuildInfo() {
            return "fake CUDA build";
        }
    }
}
