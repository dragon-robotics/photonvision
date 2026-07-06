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
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Family;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Version;
import org.photonvision.common.configuration.NeuralNetworkModelsSettings.ModelProperties;
import org.photonvision.vision.objects.Model;
import org.photonvision.vision.objects.NullModel;
import org.photonvision.vision.objects.ObjectDetector;

public class NeuralNetworkModelManagerTest {
    @Test
    void shippedPropertiesIncludeRubikAprilTagModelOnly() throws Exception {
        var manager = NeuralNetworkModelManager.getInstance(true);
        var modelsDir = new File("build/test-models");

        Method method =
                NeuralNetworkModelManager.class.getDeclaredMethod("getShippedProperties", File.class);
        method.setAccessible(true);
        var settings = (NeuralNetworkModelsSettings) method.invoke(manager, modelsDir);

        var aprilTagModels =
                List.of(settings.getModels()).stream()
                        .filter(model -> model.nickname().equals("AprilTag V4"))
                        .toList();

        assertEquals(1, aprilTagModels.size());
        var aprilTagModel = aprilTagModels.get(0);
        assertEquals(
                Path.of(modelsDir.getAbsolutePath(), "apriltagV4-yolo11.tflite"),
                aprilTagModel.modelPath());
        assertEquals(List.of("AprilTag"), aprilTagModel.labels());
        assertEquals(640, aprilTagModel.resolutionWidth());
        assertEquals(640, aprilTagModel.resolutionHeight());
        assertEquals(Family.RUBIK, aprilTagModel.family());
        assertEquals(Version.YOLOV11, aprilTagModel.version());

        assertTrue(
                List.of(settings.getModels()).stream()
                        .noneMatch(
                                model ->
                                        model.modelPath()
                                                .toString()
                                                .endsWith("apriltagV4-yolo11.rknn")));
    }

    @Test
    void getDefaultAprilTagModelDoesNotChangeObjectDetectionDefault() {
        var manager = NeuralNetworkModelManager.getInstance(true);
        manager.supportedBackends.clear();
        manager.supportedBackends.add(Family.RUBIK);

        var fuelModel =
                new StubModel(
                        new ModelProperties(
                                Path.of("fuelV1-yolo11n.tflite"),
                                "Fuel v11n",
                                List.of("Fuel"),
                                640,
                                640,
                                Family.RUBIK,
                                Version.YOLOV11));
        var aprilTagModel =
                new StubModel(
                        new ModelProperties(
                                Path.of("apriltagV4-yolo11.tflite"),
                                "AprilTag V4",
                                List.of("AprilTag"),
                                640,
                                640,
                                Family.RUBIK,
                                Version.YOLOV11));

        manager.models = new HashMap<>();
        manager.models.put(Family.RUBIK, new ArrayList<>(List.of(fuelModel, aprilTagModel)));

        assertSame(fuelModel, manager.getDefaultModel().orElseThrow());
        assertSame(aprilTagModel, manager.getDefaultAprilTagModel().orElseThrow());
    }

    @Test
    void getDefaultModelSkipsSupportedBackendsWithNoModels() {
        var manager = NeuralNetworkModelManager.getInstance(true);
        manager.supportedBackends.clear();
        manager.supportedBackends.add(Family.RUBIK);
        manager.supportedBackends.add(Family.RKNN);

        var rknnModel =
                new StubModel(
                        new ModelProperties(
                                Path.of("fuelV1-yolo11n.rknn"),
                                "Fuel v11n",
                                List.of("Fuel"),
                                640,
                                640,
                                Family.RKNN,
                                Version.YOLOV11));

        manager.models = new HashMap<>();
        manager.models.put(Family.RKNN, new ArrayList<>(List.of(rknnModel)));

        assertSame(rknnModel, manager.getDefaultModel().orElseThrow());
    }

    private record StubModel(ModelProperties properties) implements Model {
        @Override
        public ObjectDetector load() {
            return NullModel.getInstance();
        }

        @Override
        public String getUID() {
            return properties.modelPath().toString();
        }

        @Override
        public String getNickname() {
            return properties.nickname();
        }

        @Override
        public Family getFamily() {
            return properties.family();
        }

        @Override
        public ModelProperties getProperties() {
            return properties;
        }
    }
}
