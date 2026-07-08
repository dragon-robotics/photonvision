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
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.opencv.Releasable;
import org.photonvision.vision.pipe.CVPipe;

/**
 * Composite pipe for ML-assisted AprilTag detection: runs YOLO ROI detection on the input frame,
 * pads each detected ROI, then runs the traditional WPILib AprilTag decoder on the matching regions
 * of the processed (single-channel) frame.
 *
 * <p>Each stage receives a different frame because each has different input requirements: the YOLO
 * model expects the full-fidelity source image matching its training distribution, while the WPILib
 * decoder requires a single-channel mat and performs its own adaptive thresholding internally.
 */
public class AprilTagMLHybridPipe
        extends CVPipe<Frame, MLDetectionResult, AprilTagMLHybridPipe.Params> implements Releasable {
    private final AprilTagROIDetectionPipe roiDetectionPipe;
    private final AprilTagROIDecodePipe roiDecodePipe;

    private long lastRoiDetectionNanos;
    private long lastRoiExpansionNanos;
    private long lastRoiDecodeNanos;
    private int lastRawRoiCount;
    private int lastExpandedRoiCount;
    private boolean lastFreshMlFrame;
    private boolean lastUsedCachedRois;
    private boolean lastForcedRefresh;
    private boolean lastDecodeDrop;
    private List<RotatedRect> cachedExpandedRois = List.of();
    private int framesUntilMlRefresh;
    private boolean forceMlRefreshNextFrame = true;
    private int lastFreshDecodeCount;

    public static class Params {
        public final AprilTagROIDetectionPipe.AprilTagROIDetectionParams detectionParams;
        public final AprilTagROIDecodePipe.ROIDecodeParams decodeParams;
        public final int roiPaddingPixels;
        public final int mlRefreshIntervalFrames;

        public Params(
                AprilTagROIDetectionPipe.AprilTagROIDetectionParams detectionParams,
                AprilTagROIDecodePipe.ROIDecodeParams decodeParams,
                int roiPaddingPixels) {
            this(detectionParams, decodeParams, roiPaddingPixels, 1);
        }

        public Params(
                AprilTagROIDetectionPipe.AprilTagROIDetectionParams detectionParams,
                AprilTagROIDecodePipe.ROIDecodeParams decodeParams,
                int roiPaddingPixels,
                int mlRefreshIntervalFrames) {
            this.detectionParams = detectionParams;
            this.decodeParams = decodeParams;
            this.roiPaddingPixels = roiPaddingPixels;
            this.mlRefreshIntervalFrames = Math.max(1, mlRefreshIntervalFrames);
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
        if (shouldResetCache(newParams)) {
            resetRoiCache();
        }
        roiDetectionPipe.setParams(newParams.detectionParams);
        roiDecodePipe.setParams(newParams.decodeParams);
        super.setParams(newParams);
    }

    private boolean shouldResetCache(Params newParams) {
        if (params == null || newParams == null) {
            return true;
        }

        var oldDetectionParams = params.detectionParams;
        var newDetectionParams = newParams.detectionParams;
        if (oldDetectionParams == null || newDetectionParams == null) {
            return oldDetectionParams != newDetectionParams;
        }

        return params.roiPaddingPixels != newParams.roiPaddingPixels
                || params.mlRefreshIntervalFrames != newParams.mlRefreshIntervalFrames
                || oldDetectionParams.model != newDetectionParams.model
                || Double.compare(
                                oldDetectionParams.confidenceThreshold,
                                newDetectionParams.confidenceThreshold)
                        != 0
                || Double.compare(oldDetectionParams.nmsThreshold, newDetectionParams.nmsThreshold) != 0;
    }

    private void resetRoiCache() {
        cachedExpandedRois = List.of();
        framesUntilMlRefresh = 0;
        forceMlRefreshNextFrame = true;
        lastFreshDecodeCount = 0;
    }

    private static RotatedRect copyRoi(RotatedRect roi) {
        return new RotatedRect(
                new Point(roi.center.x, roi.center.y),
                new Size(roi.size.width, roi.size.height),
                roi.angle);
    }

    private static List<RotatedRect> copyRois(List<RotatedRect> rois) {
        var copiedRois = new ArrayList<RotatedRect>(rois.size());
        for (var roi : rois) {
            copiedRois.add(copyRoi(roi));
        }
        return copiedRois;
    }

    @Override
    protected MLDetectionResult process(Frame frame) {
        lastFreshMlFrame = false;
        lastUsedCachedRois = false;
        lastForcedRefresh = false;
        lastDecodeDrop = false;
        lastRawRoiCount = 0;
        lastExpandedRoiCount = 0;
        lastRoiExpansionNanos = 0;
        lastRoiDecodeNanos = 0;

        List<RotatedRect> expandedRois;
        boolean runMl = shouldRunMlRefresh();

        if (runMl) {
            lastFreshMlFrame = true;
            lastForcedRefresh = forceMlRefreshNextFrame && !cachedExpandedRois.isEmpty();
            forceMlRefreshNextFrame = false;
            framesUntilMlRefresh = params != null ? params.mlRefreshIntervalFrames - 1 : 0;

            CVPipe.CVPipeResult<List<RotatedRect>> mlResult = roiDetectionPipe.run(frame.colorImage);
            lastRoiDetectionNanos = mlResult.nanosElapsed;
            List<RotatedRect> rawRois = mlResult.output;
            lastRawRoiCount = rawRois.size();

            if (rawRois.isEmpty()) {
                cachedExpandedRois = List.of();
                lastFreshDecodeCount = 0;
                framesUntilMlRefresh = 0;
                return new MLDetectionResult(new ArrayList<>(), List.of());
            }

            int frameWidth = frame.colorImage.getMat().cols();
            int frameHeight = frame.colorImage.getMat().rows();
            expandedRois = new ArrayList<>(rawRois.size());
            long expansionStartNanos = System.nanoTime();
            for (RotatedRect roi : rawRois) {
                expandedRois.add(
                        AprilTagROIDecodePipe.expandBbox(
                                roi, params.roiPaddingPixels, frameWidth, frameHeight));
            }
            lastRoiExpansionNanos = System.nanoTime() - expansionStartNanos;
            cachedExpandedRois = copyRois(expandedRois);
        } else {
            lastRoiDetectionNanos = 0;
            lastUsedCachedRois = true;
            framesUntilMlRefresh--;
            expandedRois = copyRois(cachedExpandedRois);
        }
        lastExpandedRoiCount = expandedRois.size();

        AprilTagROIDecodePipe.ROIDecodeInput decodeInput =
                new AprilTagROIDecodePipe.ROIDecodeInput(frame.processedImage, expandedRois);

        CVPipe.CVPipeResult<List<AprilTagDetection>> decodeResult = roiDecodePipe.run(decodeInput);
        lastRoiDecodeNanos = decodeResult.nanosElapsed;
        if (lastFreshMlFrame) {
            lastFreshDecodeCount = decodeResult.output.size();
        } else if (decodeResult.output.size() < lastFreshDecodeCount) {
            lastDecodeDrop = true;
            forceMlRefreshNextFrame = true;
        }

        return new MLDetectionResult(decodeResult.output, expandedRois);
    }

    private boolean shouldRunMlRefresh() {
        if (params == null || params.mlRefreshIntervalFrames <= 1) {
            return true;
        }
        return cachedExpandedRois.isEmpty() || forceMlRefreshNextFrame || framesUntilMlRefresh <= 0;
    }

    public long getLastRoiDetectionNanos() {
        return lastRoiDetectionNanos;
    }

    public long getLastRoiExpansionNanos() {
        return lastRoiExpansionNanos;
    }

    public long getLastRoiDecodeNanos() {
        return lastRoiDecodeNanos;
    }

    public int getLastRawRoiCount() {
        return lastRawRoiCount;
    }

    public int getLastExpandedRoiCount() {
        return lastExpandedRoiCount;
    }

    public boolean getLastFreshMlFrame() {
        return lastFreshMlFrame;
    }

    public boolean getLastUsedCachedRois() {
        return lastUsedCachedRois;
    }

    public boolean getLastForcedRefresh() {
        return lastForcedRefresh;
    }

    public boolean getLastDecodeDrop() {
        return lastDecodeDrop;
    }

    /**
     * @return true when the ROI detection stage has a loaded model (not {@code NullModel}).
     */
    public boolean isAvailable() {
        return roiDetectionPipe.isAvailable();
    }

    @Override
    public void release() {
        roiDetectionPipe.release();
        roiDecodePipe.release();
    }
}
