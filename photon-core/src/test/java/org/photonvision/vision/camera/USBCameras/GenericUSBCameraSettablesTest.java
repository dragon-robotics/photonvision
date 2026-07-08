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

package org.photonvision.vision.camera.USBCameras;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import edu.wpi.first.cscore.VideoMode;
import edu.wpi.first.util.PixelFormat;
import java.util.List;
import org.junit.jupiter.api.Test;

public class GenericUSBCameraSettablesTest {
    @Test
    public void sortVideoModesForDisplayPrefersYuyvBeforeMjpegForMatchingResolutionAndFps() {
        var mjpegFirstModes =
                List.of(
                        new VideoMode(PixelFormat.kMJPEG, 1600, 1304, 60),
                        new VideoMode(PixelFormat.kMJPEG, 1280, 720, 60),
                        new VideoMode(PixelFormat.kMJPEG, 640, 480, 60),
                        new VideoMode(PixelFormat.kYUYV, 1600, 1304, 60),
                        new VideoMode(PixelFormat.kYUYV, 1280, 720, 60),
                        new VideoMode(PixelFormat.kYUYV, 640, 480, 60));
        var yuyvFirstModes =
                List.of(
                        new VideoMode(PixelFormat.kYUYV, 1600, 1304, 60),
                        new VideoMode(PixelFormat.kYUYV, 1280, 720, 60),
                        new VideoMode(PixelFormat.kYUYV, 640, 480, 60),
                        new VideoMode(PixelFormat.kMJPEG, 1600, 1304, 60),
                        new VideoMode(PixelFormat.kMJPEG, 1280, 720, 60),
                        new VideoMode(PixelFormat.kMJPEG, 640, 480, 60));

        assertHighResolutionYuyvComesBeforeMjpeg(mjpegFirstModes);
        assertHighResolutionYuyvComesBeforeMjpeg(yuyvFirstModes);
    }

    @Test
    public void didSetVideoModeFailMatchesCscoreSuccessReturnValue() {
        assertFalse(GenericUSBCameraSettables.didSetVideoModeFail(true));
        assertTrue(GenericUSBCameraSettables.didSetVideoModeFail(false));
    }

    private static void assertHighResolutionYuyvComesBeforeMjpeg(List<VideoMode> unsortedModes) {
        var sortedModes = GenericUSBCameraSettables.sortVideoModesForDisplay(unsortedModes);

        assertEquals(6, sortedModes.size());
        assertEquals(640, sortedModes.get(0).width);
        assertEquals(PixelFormat.kYUYV, sortedModes.get(0).pixelFormat);
        assertEquals(PixelFormat.kMJPEG, sortedModes.get(1).pixelFormat);
        assertEquals(1280, sortedModes.get(2).width);
        assertEquals(PixelFormat.kYUYV, sortedModes.get(2).pixelFormat);
        assertEquals(PixelFormat.kMJPEG, sortedModes.get(3).pixelFormat);
        assertEquals(1600, sortedModes.get(4).width);
        assertEquals(PixelFormat.kYUYV, sortedModes.get(4).pixelFormat);
        assertEquals(PixelFormat.kMJPEG, sortedModes.get(5).pixelFormat);
    }
}
