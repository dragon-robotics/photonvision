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

import java.awt.Color;
import java.lang.ref.Cleaner;
import java.lang.ref.Cleaner.Cleanable;
import java.nio.file.Path;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.ColorHelper;
import org.photonvision.vision.pipe.impl.NeuralNetworkPipeResult;

public class TensorRtObjectDetector implements ObjectDetector {
    private static final Logger logger = new Logger(TensorRtObjectDetector.class, LogGroup.General);

    private final Cleaner cleaner = Cleaner.create();

    private final Cleanable cleanable;

    private static Runnable cleanupAction(long ptr) {
        return () -> TensorRtJNI.destroy(ptr);
    }

    private long ptr;

    private final TensorRtModel model;

    private final Size inputSize;

    @Override
    public TensorRtModel getModel() {
        return model;
    }

    public TensorRtObjectDetector(TensorRtModel model, Size inputSize) {
        this.model = model;
        this.inputSize = inputSize;

        if (!TensorRtJNI.isAvailable()) {
            throw new UnsupportedOperationException(
                    "TensorRT JNI unavailable for model " + model.modelFile.getName());
        }

        Path enginePath =
                model.modelFile.toPath()
                        .resolveSibling(model.modelFile.getName().replaceFirst("\\.[^.]+$", ".engine"));
        ptr =
                TensorRtJNI.create(
                        model.modelFile.getPath(),
                        enginePath.toString(),
                        (int) inputSize.width,
                        (int) inputSize.height,
                        model.properties.labels().size());

        if (ptr == 0) {
            throw new UnsupportedOperationException(
                    "Failed to create TensorRT detector for model " + model.modelFile.getName());
        }

        logger.debug("Created detector for model " + model.modelFile.getName());
        cleanable = cleaner.register(this, cleanupAction(ptr));
    }

    @Override
    public List<String> getClasses() {
        return model.properties.labels();
    }

    @Override
    public synchronized List<NeuralNetworkPipeResult> detect(
            Mat in, double nmsThresh, double boxThresh) {
        if (ptr == 0) {
            logger.error("Detector is not initialized! Model: " + model.modelFile.getName());
            return List.of();
        }

        Mat letterboxed = new Mat();
        Letterbox scale =
                Letterbox.letterbox(in, letterboxed, this.inputSize, ColorHelper.colorToScalar(Color.GRAY));
        if (!letterboxed.size().equals(this.inputSize)) {
            letterboxed.release();
            throw new RuntimeException("Letterboxed frame is not the right size!");
        }

        long currentPtr = ptr;
        var results =
                TensorRtJNI.detect(currentPtr, letterboxed.getNativeObjAddr(), nmsThresh, boxThresh);
        letterboxed.release();

        if (results == null) {
            return List.of();
        }

        return scale.resizeDetections(
                List.of(results).stream()
                        .map(it -> new NeuralNetworkPipeResult(it.rect(), it.classId(), it.confidence()))
                        .toList());
    }

    @Override
    public synchronized void release() {
        cleanable.clean();
        ptr = 0;
    }
}
