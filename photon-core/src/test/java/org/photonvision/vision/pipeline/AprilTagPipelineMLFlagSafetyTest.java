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
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Family;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Version;
import org.photonvision.common.configuration.NeuralNetworkModelsSettings.ModelProperties;
import org.photonvision.common.util.TestUtils;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.provider.FileFrameProvider;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.photonvision.vision.target.TargetModel;

public class AprilTagPipelineMLFlagSafetyTest {
    @BeforeEach
    public void setup() {
        LoadJNI.loadLibraries();
        ConfigManager.getInstance().load();
    }

    @Test
    public void testMLSettingsEquality() {
        var settings1 = new AprilTagPipelineSettings();
        var settings2 = new AprilTagPipelineSettings();

        assertEquals(settings1, settings2);

        settings1.useMLDetection = true;
        assertNotEquals(settings1, settings2);

        settings2.useMLDetection = true;
        assertEquals(settings1, settings2);

        settings1.mlConfidenceThreshold = 0.7;
        assertNotEquals(settings1, settings2);

        settings2.mlConfidenceThreshold = 0.7;
        assertEquals(settings1, settings2);

        var customModel =
                new ModelProperties(
                        Path.of("test", "custom-model.tflite").toAbsolutePath(),
                        "custom-model",
                        List.of(),
                        640,
                        640,
                        Family.RUBIK,
                        Version.YOLOV11);
        settings1.model = customModel;
        assertNotEquals(settings1, settings2);

        settings2.model =
                new ModelProperties(
                        Path.of("test", "custom-model.tflite").toAbsolutePath(),
                        "custom-model",
                        List.of(),
                        640,
                        640,
                        Family.RUBIK,
                        Version.YOLOV11);
        assertEquals(settings1, settings2);
    }

    @Test
    public void testMLEnabledOnNonNpuPlatformFallsBack() {
        var pipeline = new AprilTagPipeline();

        pipeline.getSettings().useMLDetection = true;
        pipeline.getSettings().inputShouldShow = true;
        pipeline.getSettings().outputShouldDraw = true;
        pipeline.getSettings().solvePNPEnabled = true;
        pipeline.getSettings().targetModel = TargetModel.kAprilTag6p5in_36h11;
        pipeline.getSettings().tagFamily = AprilTagFamily.kTag36h11;

        var frameProvider =
                new FileFrameProvider(
                        TestUtils.getApriltagImagePath(
                                TestUtils.ApriltagTestImages.kTag1_640_480, false),
                        TestUtils.WPI2020Image.FOV,
                        TestUtils.get2020LifeCamCoeffs(false));
        frameProvider.requestFrameThresholdType(pipeline.getThresholdType());

        CVPipelineResult pipelineResult = pipeline.run(frameProvider.get(), QuirkyCamera.DefaultCamera);

        assertFalse(pipelineResult.targets.isEmpty());
        assertEquals(1, pipelineResult.targets.size());
        assertEquals(1, pipelineResult.targets.get(0).getFiducialId());

        pipelineResult.release();
        frameProvider.release();
        pipeline.release();
    }

    @Test
    public void testTraditionalDetectionWhenMLDisabled() {
        var pipeline = new AprilTagPipeline();

        pipeline.getSettings().useMLDetection = false;
        pipeline.getSettings().inputShouldShow = true;
        pipeline.getSettings().outputShouldDraw = true;
        pipeline.getSettings().solvePNPEnabled = true;
        pipeline.getSettings().targetModel = TargetModel.kAprilTag6p5in_36h11;
        pipeline.getSettings().tagFamily = AprilTagFamily.kTag36h11;

        var frameProvider =
                new FileFrameProvider(
                        TestUtils.getApriltagImagePath(
                                TestUtils.ApriltagTestImages.kTag1_640_480, false),
                        TestUtils.WPI2020Image.FOV,
                        TestUtils.get2020LifeCamCoeffs(false));
        frameProvider.requestFrameThresholdType(pipeline.getThresholdType());

        CVPipelineResult pipelineResult = pipeline.run(frameProvider.get(), QuirkyCamera.DefaultCamera);

        assertFalse(pipelineResult.targets.isEmpty());
        assertEquals(1, pipelineResult.targets.size());
        assertNotNull(pipelineResult.targets.get(0).getBestCameraToTarget3d());

        pipelineResult.release();
        frameProvider.release();
        pipeline.release();
    }
}
