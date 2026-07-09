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
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Family;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Version;
import org.photonvision.common.configuration.NeuralNetworkModelsSettings.ModelProperties;
import org.photonvision.vision.objects.TensorRtModel;

public class NeuralNetworkModelManagerTest {
    @TempDir Path tmpDir;

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

    @Test
    void loadModelCreatesTensorRtModelForSupportedBackend() throws Exception {
        ConfigManager.INSTANCE = new ConfigManager(tmpDir, new SqlConfigProvider(tmpDir));
        ConfigManager.getInstance().load();
        ConfigManager.getInstance().setWriteTaskEnabled(false);

        try {
            Path modelPath = tmpDir.resolve("models").resolve("apriltag-640-640-yolo.onnx");
            Files.createDirectories(modelPath.getParent());
            Files.createFile(modelPath);

            var properties =
                    new ModelProperties(
                            modelPath,
                            "AprilTag ROI",
                            Arrays.asList("AprilTag"),
                            640,
                            640,
                            Family.TENSORRT,
                            Version.YOLOV8);
            ConfigManager.getInstance()
                    .getConfig()
                    .neuralNetworkPropertyManager()
                    .addModelProperties(properties);

            NeuralNetworkModelManager manager = NeuralNetworkModelManager.getInstance(true);
            manager.supportedBackends.clear();
            manager.supportedBackends.add(Family.TENSORRT);

            Method loadModelMethod =
                    NeuralNetworkModelManager.class.getDeclaredMethod("loadModel", Path.class);
            loadModelMethod.setAccessible(true);
            loadModelMethod.invoke(manager, modelPath);

            assertNotNull(manager.models.get(Family.TENSORRT));
            assertEquals(1, manager.models.get(Family.TENSORRT).size());
            assertTrue(manager.models.get(Family.TENSORRT).get(0) instanceof TensorRtModel);
        } finally {
            ConfigManager.INSTANCE = null;
        }
    }
}
