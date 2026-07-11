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

package org.photonvision.jni;

import edu.wpi.first.apriltag.AprilTagDetection;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;

/** JNI facade for the Orin CUDA AprilTag detector. */
public final class GpuDetectorJNI {
    private static final Logger logger = new Logger(GpuDetectorJNI.class, LogGroup.VisionModule);

    private static boolean loadAttempted;
    private static boolean available;
    private static Throwable loadError;

    private GpuDetectorJNI() {}

    /** Returns whether the native detector library was loaded successfully. */
    public static synchronized boolean isAvailable() {
        ensureLoaded();
        return available;
    }

    /** Returns the cached library-load failure, or {@code null} when loading succeeded. */
    public static synchronized Throwable getLoadError() {
        ensureLoaded();
        return loadError;
    }

    private static void ensureLoaded() {
        if (loadAttempted) {
            return;
        }

        loadAttempted = true;
        try {
            System.loadLibrary("971apriltag");
            available = true;
            try {
                logger.info("Loaded CUDA AprilTag detector: " + getBuildInfo());
            } catch (RuntimeException | LinkageError error) {
                logger.warn("Unable to read CUDA AprilTag detector build information: " + error);
            }
        } catch (UnsatisfiedLinkError | SecurityException error) {
            loadError = error;
        }
    }

    public static native long createGpuDetector(int width, int height, int decimate);

    public static native void setCalibration(
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

    public static native AprilTagDetection[] processGray(
            long handle, long dataAddress, int width, int height, long strideBytes);

    public static native void destroyGpuDetector(long handle);

    public static native long getCudaFreeMemoryBytes();

    public static native long getCudaTotalMemoryBytes();

    public static native String getBuildInfo();
}
