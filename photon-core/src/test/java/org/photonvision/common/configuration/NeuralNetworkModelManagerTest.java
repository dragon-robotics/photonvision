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

package org.photonvision.common.configuration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.io.File;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Family;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Version;
import org.photonvision.common.configuration.NeuralNetworkModelsSettings.ModelProperties;

public class NeuralNetworkModelManagerTest {
    @Test
    void tensorRtFamilyUsesOnnxExtension() {
        assertEquals(".onnx", Family.TENSORRT.extension());
    }

    @Test
    void shippedModelsIncludeAprilTagTensorRtModel() throws Exception {
        NeuralNetworkModelManager manager = NeuralNetworkModelManager.getInstance(true);
        Method getShippedPropertiesMethod =
                NeuralNetworkModelManager.class.getDeclaredMethod("getShippedProperties", File.class);
        getShippedPropertiesMethod.setAccessible(true);

        File modelsDirectory = Path.of("build", "tmp", "task4-models").toAbsolutePath().toFile();
        NeuralNetworkModelsSettings shippedProperties =
                (NeuralNetworkModelsSettings)
                        getShippedPropertiesMethod.invoke(manager, modelsDirectory);

        ModelProperties aprilTagModel =
                Arrays.stream(shippedProperties.getModels())
                        .filter(model -> model.family() == Family.TENSORRT)
                        .findFirst()
                        .orElse(null);

        assertNotNull(aprilTagModel);
        assertEquals(
                Path.of(modelsDirectory.getAbsolutePath(), "apriltag-640-640-yolo.onnx"),
                aprilTagModel.modelPath());
        assertEquals("AprilTag ROI", aprilTagModel.nickname());
        assertEquals(Arrays.asList("AprilTag"), aprilTagModel.labels());
        assertEquals(640, aprilTagModel.resolutionWidth());
        assertEquals(640, aprilTagModel.resolutionHeight());
        assertEquals(Version.YOLOV8, aprilTagModel.version());
    }
}
