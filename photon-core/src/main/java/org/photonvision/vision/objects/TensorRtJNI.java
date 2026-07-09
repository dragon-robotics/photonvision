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

package org.photonvision.vision.objects;

import org.opencv.core.Rect2d;

public final class TensorRtJNI {
    private TensorRtJNI() {}

    public record Detection(Rect2d rect, int classId, double confidence) {}

    public static boolean isAvailable() {
        try {
            System.loadLibrary("photon_tensorrt_jni");
            return true;
        } catch (UnsatisfiedLinkError error) {
            return false;
        }
    }

    public static native long create(
            String onnxPath, String engineCachePath, int width, int height, int classCount);

    public static native Detection[] detect(long ptr, long matAddr, double nmsThresh, double boxThresh);

    public static native void destroy(long ptr);
}
