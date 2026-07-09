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

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.RotatedRect;
import org.photonvision.vision.objects.Model;
import org.photonvision.vision.objects.NullModel;
import org.photonvision.vision.objects.ObjectDetector;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.opencv.Releasable;
import org.photonvision.vision.pipe.CVPipe;

public class AprilTagROIDetectionPipe
        extends CVPipe<CVMat, List<RotatedRect>, AprilTagROIDetectionPipe.AprilTagROIDetectionParams>
        implements Releasable {
    private ObjectDetector detector;
    private Model currentModel;

    public static class AprilTagROIDetectionParams {
        public final Model model;
        public final double confidenceThreshold;
        public final double nmsThreshold;

        public AprilTagROIDetectionParams(Model model, double confidenceThreshold, double nmsThreshold) {
            this.model = model;
            this.confidenceThreshold = confidenceThreshold;
            this.nmsThreshold = nmsThreshold;
        }
    }

    public AprilTagROIDetectionPipe() {
        detector = NullModel.getInstance();
    }

    @Override
    protected List<RotatedRect> process(CVMat in) {
        List<RotatedRect> rois = new ArrayList<>();

        if (detector == null || detector instanceof NullModel) {
            return rois;
        }

        if (in.getMat().empty()) {
            return rois;
        }

        List<NeuralNetworkPipeResult> detections =
                detector.detect(in.getMat(), params.nmsThreshold, params.confidenceThreshold);

        for (NeuralNetworkPipeResult detection : detections) {
            rois.add(detection.bbox());
        }

        return rois;
    }

    @Override
    public void setParams(AprilTagROIDetectionParams newParams) {
        if (newParams.model == null) {
            if (detector != null && !(detector instanceof NullModel)) {
                detector.release();
            }
            detector = NullModel.getInstance();
            currentModel = null;
        } else if (newParams.model != currentModel) {
            if (detector != null && !(detector instanceof NullModel)) {
                detector.release();
            }
            detector = newParams.model.load();
            currentModel = newParams.model;
        }
        super.setParams(newParams);
    }

    public boolean isAvailable() {
        return detector != null && !(detector instanceof NullModel);
    }

    @Override
    public void release() {
        if (detector != null && !(detector instanceof NullModel)) {
            detector.release();
            detector = null;
        }
    }
}
