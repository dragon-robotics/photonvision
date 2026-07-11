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

import edu.wpi.first.apriltag.AprilTagDetection;
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.jni.GpuDetectorJNI;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.opencv.Releasable;
import org.photonvision.vision.pipe.CVPipe;

/** Owns the lifetime of one CUDA AprilTag detector handle. */
public class AprilTagDetectionCudaPipe
        extends CVPipe<
                CVMat, List<AprilTagDetection>, AprilTagDetectionCudaPipe.AprilTagDetectionCudaPipeParams>
        implements Releasable {
    private static final Logger logger =
            new Logger(AprilTagDetectionCudaPipe.class, LogGroup.VisionModule);

    private static final int NATIVE_ALIGNMENT = 8;

    /** Native operations used by this pipe. Tests supply a fake implementation. */
    public interface Backend {
        boolean isAvailable();

        String getLoadError();

        long createGpuDetector(int width, int height, int decimate);

        void setCalibration(
                long handle,
                double fx,
                double fy,
                double cx,
                double cy,
                double k1,
                double k2,
                double p1,
                double p2,
                double k3);

        AprilTagDetection[] processGray(
                long handle, long dataAddress, int width, int height, long strideBytes);

        void destroyGpuDetector(long handle);

        long getCudaFreeMemoryBytes();

        long getCudaTotalMemoryBytes();

        String getBuildInfo();
    }

    public record AprilTagDetectionCudaPipeParams(int decimate) {}

    public record Calibration(
            double fx,
            double fy,
            double cx,
            double cy,
            double k1,
            double k2,
            double p1,
            double p2,
            double k3) {
        private static Calibration identity() {
            return new Calibration(1, 1, 0, 0, 0, 0, 0, 0, 0);
        }
    }

    private enum State {
        IDLE,
        ACTIVE,
        FAILED,
        RELEASED
    }

    private final Backend backend;
    private boolean enabled;
    private boolean cleanupFailed;
    private State state = State.IDLE;
    private long handle;
    private int activeWidth;
    private int activeHeight;
    private int activeNativeDecimate;
    private int activeSourceWidth;
    private int activeSourceHeight;
    private boolean activeUsesJavaResize;
    private Calibration calibration = Calibration.identity();
    private Calibration appliedCalibration;

    public AprilTagDetectionCudaPipe() {
        this(new JniBackend());
    }

    public AprilTagDetectionCudaPipe(Backend backend) {
        this.backend = backend;
        this.params = new AprilTagDetectionCudaPipeParams(1);
    }

    /** Enables lazy detector construction or destroys the active detector immediately. */
    public void setEnabled(boolean enabled) {
        if (state == State.RELEASED) {
            return;
        }

        if (enabled) {
            if (!cleanupFailed) {
                this.enabled = true;
            }
            return;
        }

        this.enabled = false;
        if (cleanupFailed) {
            return;
        }

        try {
            destroyCurrentHandle();
            state = State.IDLE;
        } catch (RuntimeException | LinkageError error) {
            markCleanupFailure("Failed to disable CUDA AprilTag detector", error);
        }
    }

    /** Returns whether this enabled pipe can be selected for CUDA detection. */
    public boolean isAvailable() {
        return enabled
                && !cleanupFailed
                && state != State.FAILED
                && state != State.RELEASED
                && backend.isAvailable();
    }

    /** Updates native camera calibration without reapplying an identical value. */
    public void setCalibration(Calibration calibration) {
        if (state == State.RELEASED || cleanupFailed || calibration.equals(this.calibration)) {
            return;
        }

        this.calibration = calibration;
        if (state == State.ACTIVE) {
            try {
                applyCalibrationIfNeeded();
            } catch (RuntimeException | LinkageError error) {
                failDetection("Failed to update CUDA AprilTag calibration", error);
            }
        }
    }

    @Override
    public void setParams(AprilTagDetectionCudaPipeParams newParams) {
        if (state == State.RELEASED || cleanupFailed) {
            return;
        }

        var normalizedParams =
                new AprilTagDetectionCudaPipeParams(normalizeDecimate(newParams.decimate()));
        if (params != null
                && nativeDecimate(params.decimate()) != nativeDecimate(normalizedParams.decimate())
                && state == State.ACTIVE) {
            try {
                destroyCurrentHandle();
                state = State.IDLE;
            } catch (RuntimeException | LinkageError error) {
                markCleanupFailure("Failed to reconfigure CUDA AprilTag detector", error);
                return;
            }
        }

        super.setParams(normalizedParams);
    }

    @Override
    protected List<AprilTagDetection> process(CVMat in) {
        if (!isAvailable()) {
            return List.of();
        }

        CVMat resizedInput = null;
        try {
            Mat input = in.getMat();
            if (input.empty() || input.type() != CvType.CV_8UC1) {
                failDetection("CUDA AprilTag detector requires a nonempty CV_8UC1 image", null);
                return List.of();
            }

            var detectionInput = createDetectionInput(in, input);
            resizedInput = detectionInput.ownedResize();
            if (!ensureDetector(detectionInput)) {
                return List.of();
            }

            AprilTagDetection[] detections =
                    backend.processGray(
                            handle,
                            detectionInput.mat().dataAddr(),
                            detectionInput.mat().cols(),
                            detectionInput.mat().rows(),
                            detectionInput.mat().step1());
            if (detections == null) {
                failDetection("CUDA AprilTag detector returned a null detection array", null);
                return List.of();
            }

            return detectionInput.usesJavaResize()
                    ? scaleDetections(detections, detectionInput.xScale(), detectionInput.yScale())
                    : List.of(detections);
        } catch (RuntimeException | LinkageError error) {
            failDetection("CUDA AprilTag detection failed", error);
            return List.of();
        } finally {
            if (resizedInput != null) {
                resizedInput.release();
            }
        }
    }

    @Override
    public void release() {
        if (state == State.RELEASED) {
            return;
        }

        enabled = false;
        if (!cleanupFailed) {
            try {
                destroyCurrentHandle();
            } catch (RuntimeException | LinkageError error) {
                markCleanupFailure("Failed to release CUDA AprilTag detector", error);
            }
        }
        state = State.RELEASED;
    }

    private DetectionInput createDetectionInput(CVMat input, Mat inputMat) {
        int decimate = normalizeDecimate(params.decimate());
        if (decimate <= 2) {
            return new DetectionInput(inputMat, null, decimate, false, 1, 1);
        }

        int width = alignDown(inputMat.cols() / decimate, NATIVE_ALIGNMENT);
        int height = alignDown(inputMat.rows() / decimate, NATIVE_ALIGNMENT);
        if (width < NATIVE_ALIGNMENT || height < NATIVE_ALIGNMENT) {
            throw new IllegalArgumentException("CUDA AprilTag resized image is too small");
        }

        CVMat resizedInput = new CVMat();
        boolean ownershipTransferred = false;
        try {
            Imgproc.resize(inputMat, resizedInput.getMat(), new Size(width, height));
            var detectionInput =
                    new DetectionInput(
                            resizedInput.getMat(),
                            resizedInput,
                            1,
                            true,
                            inputMat.cols() / (double) width,
                            inputMat.rows() / (double) height);
            ownershipTransferred = true;
            return detectionInput;
        } finally {
            if (!ownershipTransferred) {
                resizedInput.release();
            }
        }
    }

    private boolean ensureDetector(DetectionInput input) {
        if (state == State.ACTIVE
                && (activeWidth != input.mat().cols()
                        || activeHeight != input.mat().rows()
                        || activeNativeDecimate != input.nativeDecimate())) {
            try {
                destroyCurrentHandle();
                state = State.IDLE;
            } catch (RuntimeException | LinkageError error) {
                markCleanupFailure("Failed to replace CUDA AprilTag detector", error);
                return false;
            }
        }

        if (state != State.IDLE) {
            return state == State.ACTIVE;
        }

        logMemory("before creation");
        try {
            handle =
                    backend.createGpuDetector(input.mat().cols(), input.mat().rows(), input.nativeDecimate());
            if (handle == 0) {
                throw new IllegalStateException("CUDA AprilTag detector returned an invalid handle");
            }
            state = State.ACTIVE;
            activeWidth = input.mat().cols();
            activeHeight = input.mat().rows();
            activeNativeDecimate = input.nativeDecimate();
            activeSourceWidth = Math.round(input.mat().cols() * (float) input.xScale());
            activeSourceHeight = Math.round(input.mat().rows() * (float) input.yScale());
            activeUsesJavaResize = input.usesJavaResize();
            appliedCalibration = null;
            logMemory("after creation");
            applyCalibrationIfNeeded();
            return true;
        } catch (RuntimeException | LinkageError error) {
            failDetection("Failed to create CUDA AprilTag detector", error);
            return false;
        }
    }

    private void applyCalibrationIfNeeded() {
        if (calibration.equals(appliedCalibration)) {
            return;
        }

        double xScale = activeUsesJavaResize ? activeWidth / (double) activeSourceWidth : 1;
        double yScale = activeUsesJavaResize ? activeHeight / (double) activeSourceHeight : 1;
        backend.setCalibration(
                handle,
                calibration.fx() * xScale,
                calibration.fy() * yScale,
                calibration.cx() * xScale,
                calibration.cy() * yScale,
                calibration.k1(),
                calibration.k2(),
                calibration.p1(),
                calibration.p2(),
                calibration.k3());
        appliedCalibration = calibration;
    }

    private void failDetection(String message, Throwable cause) {
        try {
            destroyCurrentHandle();
            state = State.FAILED;
        } catch (RuntimeException | LinkageError cleanupError) {
            markCleanupFailure(message + " and cleanup failed", cleanupError);
            return;
        }

        if (cause == null) {
            logger.error(message);
        } else {
            logger.error(message, cause);
        }
    }

    private void destroyCurrentHandle() {
        if (handle == 0) {
            return;
        }

        backend.destroyGpuDetector(handle);
        handle = 0;
        activeWidth = 0;
        activeHeight = 0;
        activeNativeDecimate = 0;
        activeSourceWidth = 0;
        activeSourceHeight = 0;
        activeUsesJavaResize = false;
        appliedCalibration = null;
        logMemory("after destruction");
    }

    private void markCleanupFailure(String message, Throwable error) {
        cleanupFailed = true;
        state = State.FAILED;
        logger.error(message, error);
    }

    private void logMemory(String stage) {
        try {
            logger.info(
                    "CUDA AprilTag memory "
                            + stage
                            + ": "
                            + backend.getCudaFreeMemoryBytes()
                            + "/"
                            + backend.getCudaTotalMemoryBytes()
                            + " bytes free");
        } catch (RuntimeException | LinkageError error) {
            logger.warn("Unable to read CUDA AprilTag memory " + stage + ": " + error);
        }
    }

    private static int normalizeDecimate(int decimate) {
        return Math.max(1, decimate);
    }

    private static int nativeDecimate(int decimate) {
        return decimate <= 2 ? decimate : 1;
    }

    private static int alignDown(int value, int alignment) {
        return value / alignment * alignment;
    }

    private static List<AprilTagDetection> scaleDetections(
            AprilTagDetection[] detections, double xScale, double yScale) {
        var scaledDetections = new ArrayList<AprilTagDetection>(detections.length);
        for (AprilTagDetection detection : detections) {
            double[] homography = detection.getHomography().clone();
            homography[0] *= xScale;
            homography[1] *= xScale;
            homography[2] *= xScale;
            homography[3] *= yScale;
            homography[4] *= yScale;
            homography[5] *= yScale;

            double[] corners = detection.getCorners().clone();
            for (int corner = 0; corner < corners.length; corner += 2) {
                corners[corner] *= xScale;
                corners[corner + 1] *= yScale;
            }

            scaledDetections.add(
                    new AprilTagDetection(
                            detection.getFamily(),
                            detection.getId(),
                            detection.getHamming(),
                            detection.getDecisionMargin(),
                            homography,
                            detection.getCenterX() * xScale,
                            detection.getCenterY() * yScale,
                            corners));
        }
        return scaledDetections;
    }

    private record DetectionInput(
            Mat mat,
            CVMat ownedResize,
            int nativeDecimate,
            boolean usesJavaResize,
            double xScale,
            double yScale) {}

    private static final class JniBackend implements Backend {
        @Override
        public boolean isAvailable() {
            return GpuDetectorJNI.isAvailable();
        }

        @Override
        public String getLoadError() {
            Throwable error = GpuDetectorJNI.getLoadError();
            return error == null ? null : error.toString();
        }

        @Override
        public long createGpuDetector(int width, int height, int decimate) {
            return GpuDetectorJNI.createGpuDetector(width, height, decimate);
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
            GpuDetectorJNI.setCalibration(handle, fx, fy, cx, cy, k1, k2, p1, p2, k3);
        }

        @Override
        public AprilTagDetection[] processGray(
                long handle, long dataAddress, int width, int height, long strideBytes) {
            return GpuDetectorJNI.processGray(handle, dataAddress, width, height, strideBytes);
        }

        @Override
        public void destroyGpuDetector(long handle) {
            GpuDetectorJNI.destroyGpuDetector(handle);
        }

        @Override
        public long getCudaFreeMemoryBytes() {
            return GpuDetectorJNI.getCudaFreeMemoryBytes();
        }

        @Override
        public long getCudaTotalMemoryBytes() {
            return GpuDetectorJNI.getCudaTotalMemoryBytes();
        }

        @Override
        public String getBuildInfo() {
            return GpuDetectorJNI.getBuildInfo();
        }
    }
}
