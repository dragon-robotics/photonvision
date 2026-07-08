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

package org.photonvision.vision.processes;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class VisionRunnerProfileOverridesTest {
    @AfterEach
    void clearProperties() {
        System.clearProperty("photonvision.profile.nonBlockFrames");
    }

    @Test
    void blockForFramesFollowsPipelineSettingByDefault() {
        System.clearProperty("photonvision.profile.nonBlockFrames");

        assertTrue(VisionRunner.shouldBlockForFrames(true));
        assertFalse(VisionRunner.shouldBlockForFrames(false));
    }

    @Test
    void profileNonBlockOverrideForcesNonBlockingFrameGrab() {
        System.setProperty("photonvision.profile.nonBlockFrames", "true");

        assertFalse(VisionRunner.shouldBlockForFrames(true));
        assertFalse(VisionRunner.shouldBlockForFrames(false));
    }
}
