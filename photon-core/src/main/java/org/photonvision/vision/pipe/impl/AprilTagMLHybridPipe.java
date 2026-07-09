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

import edu.wpi.first.apriltag.AprilTagDetection;
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.RotatedRect;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.opencv.Releasable;
import org.photonvision.vision.pipe.CVPipe;

public class AprilTagMLHybridPipe
        extends CVPipe<Frame, MLDetectionResult, AprilTagMLHybridPipe.Params> implements Releasable {
    private final AprilTagROIDetectionPipe roiDetectionPipe;
    private final AprilTagROIDecodePipe roiDecodePipe;

    public static class Params {
        public final AprilTagROIDetectionPipe.AprilTagROIDetectionParams detectionParams;
        public final AprilTagROIDecodePipe.ROIDecodeParams decodeParams;
        public final int roiPaddingPixels;

        public Params(
                AprilTagROIDetectionPipe.AprilTagROIDetectionParams detectionParams,
                AprilTagROIDecodePipe.ROIDecodeParams decodeParams,
                int roiPaddingPixels) {
            this.detectionParams = detectionParams;
            this.decodeParams = decodeParams;
            this.roiPaddingPixels = roiPaddingPixels;
        }
    }

    public AprilTagMLHybridPipe() {
        this(new AprilTagROIDetectionPipe(), new AprilTagROIDecodePipe());
    }

    AprilTagMLHybridPipe(
            AprilTagROIDetectionPipe roiDetectionPipe, AprilTagROIDecodePipe roiDecodePipe) {
        this.roiDetectionPipe = roiDetectionPipe;
        this.roiDecodePipe = roiDecodePipe;
    }

    @Override
    public void setParams(Params newParams) {
        roiDetectionPipe.setParams(newParams.detectionParams);
        roiDecodePipe.setParams(newParams.decodeParams);
        super.setParams(newParams);
    }

    @Override
    protected MLDetectionResult process(Frame frame) {
        CVPipeResult<List<RotatedRect>> detectionResult = roiDetectionPipe.run(frame.colorImage);
        List<RotatedRect> rawRois = detectionResult.output;
        if (rawRois.isEmpty()) {
            return new MLDetectionResult(new ArrayList<>(), List.of(), detectionResult.nanosElapsed);
        }

        int frameWidth = frame.colorImage.getMat().cols();
        int frameHeight = frame.colorImage.getMat().rows();
        List<RotatedRect> expandedRois = new ArrayList<>(rawRois.size());
        for (RotatedRect roi : rawRois) {
            expandedRois.add(
                    AprilTagROIDecodePipe.expandBbox(
                            roi, params.roiPaddingPixels, frameWidth, frameHeight));
        }

        var decodeInput = new AprilTagROIDecodePipe.ROIDecodeInput(frame.processedImage, expandedRois);
        CVPipeResult<List<AprilTagDetection>> decodeResult = roiDecodePipe.run(decodeInput);

        return new MLDetectionResult(
                decodeResult.output, expandedRois, detectionResult.nanosElapsed + decodeResult.nanosElapsed);
    }

    public boolean isAvailable() {
        return roiDetectionPipe.isAvailable();
    }

    @Override
    public void release() {
        roiDetectionPipe.release();
        roiDecodePipe.release();
    }
}
