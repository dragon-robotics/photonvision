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

package org.photonvision.vision.processes;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import edu.wpi.first.cscore.VideoMode;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.photonvision.common.configuration.CameraConfiguration;
import org.photonvision.vision.camera.PVCameraInfo;
import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.pipeline.CVPipelineSettings;

public class VisionModuleCameraSettingsTest {
    @Test
    void manualPipelinePrimesExposureBeforeDisablingAutoExposure() {
        var settables = new RecordingSettables();
        var settings = new CVPipelineSettings();
        settings.cameraAutoExposure = false;
        settings.cameraExposureRaw = 37;

        VisionModule.applyCameraSettingsForPipeline(settables, settings, QuirkyCamera.DefaultCamera);

        assertTrue(
                settables.calls.indexOf("exposure:37.0") < settables.calls.indexOf("autoExposure:false"));
    }

    @Test
    void autoExposurePipelineDoesNotWriteManualExposure() {
        var settables = new RecordingSettables();
        var settings = new CVPipelineSettings();
        settings.cameraAutoExposure = true;
        settings.cameraExposureRaw = 37;

        VisionModule.applyCameraSettingsForPipeline(settables, settings, QuirkyCamera.DefaultCamera);

        assertTrue(settables.calls.contains("autoExposure:true"));
        assertFalse(settables.calls.contains("exposure:37.0"));
    }

    private static class RecordingSettables extends VisionSourceSettables {
        private final List<String> calls = new ArrayList<>();

        RecordingSettables() {
            super(new CameraConfiguration(PVCameraInfo.fromFileInfo("Test", "test")));
        }

        @Override
        public void setExposureRaw(double exposureRaw) {
            calls.add("exposure:" + exposureRaw);
        }

        @Override
        public double getMinExposureRaw() {
            return 1;
        }

        @Override
        public double getMaxExposureRaw() {
            return 100;
        }

        @Override
        public void setAutoExposure(boolean cameraAutoExposure) {
            calls.add("autoExposure:" + cameraAutoExposure);
        }

        @Override
        public void setWhiteBalanceTemp(double temp) {
            calls.add("whiteBalanceTemp:" + temp);
        }

        @Override
        public void setAutoWhiteBalance(boolean autowb) {
            calls.add("autoWhiteBalance:" + autowb);
        }

        @Override
        public void setBrightness(int brightness) {
            calls.add("brightness:" + brightness);
        }

        @Override
        public void setGain(int gain) {
            calls.add("gain:" + gain);
        }

        @Override
        public VideoMode getCurrentVideoMode() {
            return new VideoMode(0, 320, 240, 30);
        }

        @Override
        protected void setVideoModeInternal(VideoMode videoMode) {
            frameStaticProperties = new FrameStaticProperties(videoMode, getFOV(), null);
            calls.add("videoMode");
        }

        @Override
        public HashMap<Integer, VideoMode> getAllVideoModes() {
            var modes = new HashMap<Integer, VideoMode>();
            modes.put(0, getCurrentVideoMode());
            return modes;
        }

        @Override
        public double getMinWhiteBalanceTemp() {
            return 1;
        }

        @Override
        public double getMaxWhiteBalanceTemp() {
            return 4000;
        }
    }
}
