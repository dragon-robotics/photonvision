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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import edu.wpi.first.apriltag.AprilTagDetection;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.photonvision.common.LoadJNI;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.opencv.CVMat;

public class AprilTagMLHybridPipeRefreshTest {
    private final List<Frame> framesToRelease = new ArrayList<>();

    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
    }

    @AfterEach
    public void releaseFrames() {
        for (var frame : framesToRelease) {
            frame.release();
        }
    }

    @Test
    public void reusesExpandedRoisBetweenMlRefreshFrames() {
        var firstRoi = roi(20, 20, 10, 10);
        var secondRoi = roi(70, 70, 10, 10);
        var detectionPipe =
                new FakeDetectionPipe(List.of(List.of(firstRoi), List.of(secondRoi)));
        var decodePipe = new FakeDecodePipe(List.of(List.of(detection(1))));
        var pipe = new AprilTagMLHybridPipe(detectionPipe, decodePipe);
        pipe.setParams(
                new AprilTagMLHybridPipe.Params(
                        null, new AprilTagROIDecodePipe.ROIDecodeParams(), 5, 2));

        pipe.run(frame());
        pipe.run(frame());
        pipe.run(frame());

        assertEquals(2, detectionPipe.runCount);
        assertEquals(3, decodePipe.inputs.size());
        assertEquals(decodePipe.inputs.get(0).get(0).center.x, decodePipe.inputs.get(1).get(0).center.x);
        assertEquals(decodePipe.inputs.get(0).get(0).center.y, decodePipe.inputs.get(1).get(0).center.y);
        assertFalse(pipe.getLastUsedCachedRois());
        assertTrue(pipe.getLastFreshMlFrame());
        assertEquals(secondRoi.center.x, decodePipe.inputs.get(2).get(0).center.x, 5.0);

        pipe.release();
    }

    @Test
    public void cachedRoiDecodeDropForcesNextMlRefresh() {
        var firstRoi = roi(20, 20, 10, 10);
        var secondRoi = roi(70, 70, 10, 10);
        var detectionPipe =
                new FakeDetectionPipe(List.of(List.of(firstRoi), List.of(secondRoi)));
        var decodePipe =
                new FakeDecodePipe(
                        List.of(
                                List.of(detection(1), detection(2)),
                                List.of(detection(1)),
                                List.of(detection(1))));
        var pipe = new AprilTagMLHybridPipe(detectionPipe, decodePipe);
        pipe.setParams(
                new AprilTagMLHybridPipe.Params(
                        null, new AprilTagROIDecodePipe.ROIDecodeParams(), 5, 4));

        pipe.run(frame());
        pipe.run(frame());
        assertTrue(pipe.getLastDecodeDrop());

        pipe.run(frame());

        assertEquals(2, detectionPipe.runCount);
        assertTrue(pipe.getLastForcedRefresh());

        pipe.release();
    }

    private Frame frame() {
        var frame =
                new Frame(
                        framesToRelease.size(),
                        new CVMat(new Mat(100, 100, CvType.CV_8UC3)),
                        new CVMat(new Mat(100, 100, CvType.CV_8UC1)),
                        FrameThresholdType.GREYSCALE,
                        new FrameStaticProperties(100, 100, 0, null));
        framesToRelease.add(frame);
        return frame;
    }

    private static RotatedRect roi(double x, double y, double width, double height) {
        return new RotatedRect(new Point(x, y), new Size(width, height), 0);
    }

    private static AprilTagDetection detection(int id) {
        return new AprilTagDetection(
                "tag36h11",
                id,
                0,
                100.0f,
                new double[] {1, 0, 0, 0, 1, 0, 0, 0, 1},
                5.0,
                5.0,
                new double[] {0, 0, 10, 0, 10, 10, 0, 10});
    }

    private static class FakeDetectionPipe extends AprilTagROIDetectionPipe {
        private final List<List<RotatedRect>> outputs;
        private int runCount;

        FakeDetectionPipe(List<List<RotatedRect>> outputs) {
            this.outputs = outputs;
        }

        @Override
        protected List<RotatedRect> process(CVMat in) {
            var output = outputs.get(Math.min(runCount, outputs.size() - 1));
            runCount++;
            return output;
        }

        @Override
        public void setParams(AprilTagROIDetectionParams newParams) {}

        @Override
        public void release() {}
    }

    private static class FakeDecodePipe extends AprilTagROIDecodePipe {
        private final List<List<AprilTagDetection>> outputs;
        private final List<List<RotatedRect>> inputs = new ArrayList<>();
        private int runCount;

        FakeDecodePipe(List<List<AprilTagDetection>> outputs) {
            this.outputs = outputs;
        }

        @Override
        protected List<AprilTagDetection> process(ROIDecodeInput input) {
            inputs.add(input.rois);
            var output = outputs.get(Math.min(runCount, outputs.size() - 1));
            runCount++;
            return output;
        }

        @Override
        public void setParams(ROIDecodeParams newParams) {}

        @Override
        public void release() {}
    }
}
