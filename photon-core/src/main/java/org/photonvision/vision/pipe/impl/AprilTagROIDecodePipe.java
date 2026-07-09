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
import edu.wpi.first.apriltag.AprilTagDetector;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.opencv.Releasable;
import org.photonvision.vision.pipe.CVPipe;

public class AprilTagROIDecodePipe
        extends CVPipe<
                AprilTagROIDecodePipe.ROIDecodeInput,
                List<AprilTagDetection>,
                AprilTagROIDecodePipe.ROIDecodeParams>
        implements Releasable {
    private static final Logger logger =
            new Logger(AprilTagROIDecodePipe.class, LogGroup.VisionModule);
    private static final boolean DEBUG_COORDINATE_MAPPING = false;

    private static class ATRContext {
        final double scaleFactor;
        final boolean wasScaled;

        ATRContext(double scaleFactor) {
            this.scaleFactor = scaleFactor;
            this.wasScaled = scaleFactor < 1.0;
        }

        static ATRContext noScaling() {
            return new ATRContext(1.0);
        }
    }

    public static class ROIDecodeInput {
        public final CVMat grayFrame;
        public final List<RotatedRect> rois;

        public ROIDecodeInput(CVMat grayFrame, List<RotatedRect> rois) {
            this.grayFrame = grayFrame;
            this.rois = rois;
        }
    }

    public static class ROIDecodeParams {
        public AprilTagFamily tagFamily = AprilTagFamily.kTag36h11;
        public AprilTagDetector.Config detectorConfig;
        public AprilTagDetector.QuadThresholdParameters quadParams;
        public int maxHammingDistance = 0;
        public double minDecisionMargin = 35;
        public boolean atrEnabled = true;
        public int atrTargetDimension = 160;
        public double atrMinScaleFactor = 0.25;

        public ROIDecodeParams() {
            detectorConfig = new AprilTagDetector.Config();
            detectorConfig.numThreads = 1;
            detectorConfig.quadDecimate = 1;
            quadParams = new AprilTagDetector.QuadThresholdParameters();
            quadParams.minClusterPixels = 5;
            quadParams.maxNumMaxima = 10;
            quadParams.criticalAngle = 45 * Math.PI / 180.0;
            quadParams.maxLineFitMSE = 10.0f;
            quadParams.minWhiteBlackDiff = 5;
            quadParams.deglitch = false;
        }
    }

    private AprilTagDetector detector;
    private AprilTagFamily currentFamily;
    private final Mat resizedRoiMat = new Mat();

    public AprilTagROIDecodePipe() {
        detector = new AprilTagDetector();
    }

    @Override
    protected List<AprilTagDetection> process(ROIDecodeInput input) {
        List<AprilTagDetection> allDetections = new ArrayList<>(input.rois.size());

        if (input.grayFrame == null || input.grayFrame.getMat().empty()) {
            return allDetections;
        }

        Mat fullFrame = input.grayFrame.getMat();
        int frameWidth = fullFrame.cols();
        int frameHeight = fullFrame.rows();

        for (RotatedRect roi : input.rois) {
            Rect roiRect = toIntRect(roi);

            if (roiRect.width <= 0 || roiRect.height <= 0) {
                continue;
            }

            roiRect = clampToFrame(roiRect, frameWidth, frameHeight);
            if (roiRect.width <= 0 || roiRect.height <= 0) {
                continue;
            }

            if (DEBUG_COORDINATE_MAPPING) {
                logger.debug(
                        "Expanded ROI x="
                                + roiRect.x
                                + " y="
                                + roiRect.y
                                + " w="
                                + roiRect.width
                                + " h="
                                + roiRect.height);
            }

            Mat roiMat = fullFrame.submat(roiRect);

            ATRContext atrContext;
            Mat processingMat;
            if (params.atrEnabled && roiRect.width > params.atrTargetDimension) {
                double scale = (double) params.atrTargetDimension / roiRect.width;
                scale = Math.max(scale, params.atrMinScaleFactor);

                int scaledWidth = (int) Math.round(roiRect.width * scale);
                int scaledHeight = (int) Math.round(roiRect.height * scale);

                atrContext = new ATRContext(scale);
                processingMat = resizedRoiMat;
                Imgproc.resize(
                        roiMat,
                        processingMat,
                        new Size(scaledWidth, scaledHeight),
                        0,
                        0,
                        Imgproc.INTER_AREA);
            } else {
                atrContext = ATRContext.noScaling();
                processingMat = roiMat;
            }

            AprilTagDetection[] roiDetections = detector.detect(processingMat);

            for (AprilTagDetection det : roiDetections) {
                if (det.getHamming() > params.maxHammingDistance) continue;
                if (det.getDecisionMargin() < params.minDecisionMargin) continue;

                allDetections.add(
                        atrContext.wasScaled
                                ? mapToFullFrameWithATR(det, roiRect, atrContext)
                                : mapToFullFrame(det, roiRect));
            }

            roiMat.release();
        }

        return deduplicateByTagId(allDetections);
    }

    private AprilTagDetection mapToFullFrame(AprilTagDetection det, Rect roiOffset) {
        double[] mappedCorners = new double[8];
        for (int i = 0; i < 4; i++) {
            mappedCorners[i * 2] = det.getCornerX(i) + roiOffset.x;
            mappedCorners[i * 2 + 1] = det.getCornerY(i) + roiOffset.y;
        }

        double centerX = det.getCenterX() + roiOffset.x;
        double centerY = det.getCenterY() + roiOffset.y;
        double[] transformedHomography =
                transformHomography(det.getHomography(), roiOffset.x, roiOffset.y);

        return new AprilTagDetection(
                det.getFamily(),
                det.getId(),
                det.getHamming(),
                det.getDecisionMargin(),
                transformedHomography,
                centerX,
                centerY,
                mappedCorners);
    }

    private AprilTagDetection mapToFullFrameWithATR(
            AprilTagDetection det, Rect roiOffset, ATRContext atrContext) {
        double invScale = 1.0 / atrContext.scaleFactor;

        double[] mappedCorners = new double[8];
        for (int i = 0; i < 4; i++) {
            mappedCorners[i * 2] = det.getCornerX(i) * invScale + roiOffset.x;
            mappedCorners[i * 2 + 1] = det.getCornerY(i) * invScale + roiOffset.y;
        }

        double centerX = det.getCenterX() * invScale + roiOffset.x;
        double centerY = det.getCenterY() * invScale + roiOffset.y;
        double[] transformedHomography =
                transformHomographyWithScale(
                        det.getHomography(), roiOffset.x, roiOffset.y, atrContext.scaleFactor);

        return new AprilTagDetection(
                det.getFamily(),
                det.getId(),
                det.getHamming(),
                det.getDecisionMargin(),
                transformedHomography,
                centerX,
                centerY,
                mappedCorners);
    }

    private double[] transformHomography(double[] homography, int offsetX, int offsetY) {
        double[] result = new double[9];
        result[0] = homography[0] + offsetX * homography[6];
        result[1] = homography[1] + offsetX * homography[7];
        result[2] = homography[2] + offsetX * homography[8];
        result[3] = homography[3] + offsetY * homography[6];
        result[4] = homography[4] + offsetY * homography[7];
        result[5] = homography[5] + offsetY * homography[8];
        result[6] = homography[6];
        result[7] = homography[7];
        result[8] = homography[8];
        return result;
    }

    private double[] transformHomographyWithScale(
            double[] homography, int offsetX, int offsetY, double scale) {
        double invScale = 1.0 / scale;
        double[] result = new double[9];
        result[0] = homography[0] * invScale + offsetX * homography[6];
        result[1] = homography[1] * invScale + offsetX * homography[7];
        result[2] = homography[2] * invScale + offsetX * homography[8];
        result[3] = homography[3] * invScale + offsetY * homography[6];
        result[4] = homography[4] * invScale + offsetY * homography[7];
        result[5] = homography[5] * invScale + offsetY * homography[8];
        result[6] = homography[6];
        result[7] = homography[7];
        result[8] = homography[8];
        return result;
    }

    public static RotatedRect expandBbox(
            RotatedRect bbox, int paddingPixels, int imageWidth, int imageHeight) {
        Rect original = bbox.boundingRect();
        int newWidth = Math.min(original.width + 2 * paddingPixels, imageWidth);
        int newHeight = Math.min(original.height + 2 * paddingPixels, imageHeight);

        int x = Math.max(0, Math.min(original.x - paddingPixels, imageWidth - newWidth));
        int y = Math.max(0, Math.min(original.y - paddingPixels, imageHeight - newHeight));

        return fromIntRect(x, y, newWidth, newHeight, bbox.angle);
    }

    private static RotatedRect fromIntRect(int x, int y, int width, int height, double angle) {
        return new RotatedRect(
                new Point(x + (width - 1) / 2.0, y + (height - 1) / 2.0),
                new Size(Math.max(0, width - 1), Math.max(0, height - 1)),
                angle);
    }

    private Rect toIntRect(RotatedRect rect) {
        return rect.boundingRect();
    }

    private Rect clampToFrame(Rect rect, int frameWidth, int frameHeight) {
        int x = Math.max(0, rect.x);
        int y = Math.max(0, rect.y);
        int w = Math.min(rect.width, frameWidth - x);
        int h = Math.min(rect.height, frameHeight - y);
        return new Rect(x, y, w, h);
    }

    private List<AprilTagDetection> deduplicateByTagId(List<AprilTagDetection> detections) {
        if (detections.size() <= 1) {
            return detections;
        }

        Map<Integer, AprilTagDetection> bestByTagId = new HashMap<>();
        for (AprilTagDetection detection : detections) {
            int tagId = detection.getId();
            AprilTagDetection existing = bestByTagId.get(tagId);
            if (existing == null || detection.getDecisionMargin() > existing.getDecisionMargin()) {
                bestByTagId.put(tagId, detection);
            }
        }

        return new ArrayList<>(bestByTagId.values());
    }

    @Override
    public void setParams(ROIDecodeParams newParams) {
        if (newParams.tagFamily != currentFamily) {
            detector.clearFamilies();
            detector.addFamily(newParams.tagFamily.getNativeName());
            currentFamily = newParams.tagFamily;
        }

        detector.setConfig(newParams.detectorConfig);
        detector.setQuadThresholdParameters(newParams.quadParams);
        super.setParams(newParams);
    }

    @Override
    public void release() {
        resizedRoiMat.release();
        if (detector != null) {
            detector.close();
            detector = null;
        }
    }
}
