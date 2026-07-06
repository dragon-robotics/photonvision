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

package org.photonvision.vision.pipe;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import edu.wpi.first.apriltag.AprilTagDetection;
import edu.wpi.first.apriltag.AprilTagDetector;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.util.TestUtils;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.frame.provider.FileFrameProvider;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipe.impl.AprilTagROIDecodePipe;
import org.photonvision.vision.pipe.impl.AprilTagROIDecodePipe.ROIDecodeInput;
import org.photonvision.vision.pipe.impl.AprilTagROIDecodePipe.ROIDecodeParams;

public class AprilTagROIDecodePipeTest {
    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
        ConfigManager.getInstance().load();
    }

    private static RotatedRect roiFromXYWH(double x, double y, double w, double h) {
        return new RotatedRect(new Point(x + w / 2.0, y + h / 2.0), new Size(w, h), 0);
    }

    @Test
    public void coordinateMappingMatchesFullFrameDetection() {
        var frameProvider =
                new FileFrameProvider(
                        TestUtils.getApriltagImagePath(TestUtils.ApriltagTestImages.kTag1_640_480, false),
                        TestUtils.WPI2020Image.FOV,
                        TestUtils.get2020LifeCamCoeffs(false));
        frameProvider.requestFrameThresholdType(FrameThresholdType.GREYSCALE);

        var frame = frameProvider.get();
        var params = new ROIDecodeParams();
        params.tagFamily = AprilTagFamily.kTag36h11;
        params.maxHammingDistance = 0;
        params.minDecisionMargin = 35;

        var fullFrameDetector = new AprilTagDetector();
        fullFrameDetector.addFamily(AprilTagFamily.kTag36h11.getNativeName());
        fullFrameDetector.setConfig(params.detectorConfig);
        fullFrameDetector.setQuadThresholdParameters(params.quadParams);

        AprilTagDetection[] fullFrameDetections =
                fullFrameDetector.detect(frame.processedImage.getMat());
        assumeTrue(fullFrameDetections.length > 0, "full-frame detector found no tag");
        var groundTruth = fullFrameDetections[0];

        double minX = Double.MAX_VALUE;
        double maxX = Double.MIN_VALUE;
        double minY = Double.MAX_VALUE;
        double maxY = Double.MIN_VALUE;
        for (int i = 0; i < 4; i++) {
            minX = Math.min(minX, groundTruth.getCornerX(i));
            maxX = Math.max(maxX, groundTruth.getCornerX(i));
            minY = Math.min(minY, groundTruth.getCornerY(i));
            maxY = Math.max(maxY, groundTruth.getCornerY(i));
        }

        var pipe = new AprilTagROIDecodePipe();
        pipe.setParams(params);

        var roi = roiFromXYWH(minX - 20, minY - 20, maxX - minX + 40, maxY - minY + 40);
        var result = pipe.run(new ROIDecodeInput(frame.processedImage, List.of(roi)));

        assertEquals(1, result.output.size());
        var mapped = result.output.get(0);
        for (int i = 0; i < 4; i++) {
            assertEquals(groundTruth.getCornerX(i), mapped.getCornerX(i), 0.5);
            assertEquals(groundTruth.getCornerY(i), mapped.getCornerY(i), 0.5);
        }
        assertEquals(groundTruth.getCenterX(), mapped.getCenterX(), 0.5);
        assertEquals(groundTruth.getCenterY(), mapped.getCenterY(), 0.5);
        assertEquals(groundTruth.getId(), mapped.getId());

        fullFrameDetector.close();
        pipe.release();
        frame.release();
        frameProvider.release();
    }

    @Test
    public void emptyRoiListReturnsEmptyResult() {
        var mat = Mat.zeros(480, 640, CvType.CV_8UC1);
        var pipe = new AprilTagROIDecodePipe();
        var params = new ROIDecodeParams();
        params.tagFamily = AprilTagFamily.kTag36h11;
        pipe.setParams(params);

        var cvMat = new CVMat(mat);
        var result = pipe.run(new ROIDecodeInput(cvMat, new ArrayList<>()));

        assertNotNull(result.output);
        assertTrue(result.output.isEmpty());
        cvMat.release();
        pipe.release();
    }

    @Test
    public void expandBboxClampsToFrame() {
        var expanded = AprilTagROIDecodePipe.expandBbox(roiFromXYWH(600, 440, 30, 30), 80, 640, 480);

        assertTrue(expanded.boundingRect().x >= 0);
        assertTrue(expanded.boundingRect().y >= 0);
        assertTrue(expanded.boundingRect().x + expanded.boundingRect().width <= 640);
        assertTrue(expanded.boundingRect().y + expanded.boundingRect().height <= 480);
        assertEquals(190, expanded.boundingRect().width);
        assertEquals(190, expanded.boundingRect().height);
    }
}
