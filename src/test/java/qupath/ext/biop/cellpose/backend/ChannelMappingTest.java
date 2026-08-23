/*-
 * Copyright 2026 QuPath developers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package qupath.ext.biop.cellpose.backend;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * The builder's {@code cellposeChannels(a, b)} carries cellpose's {@code --chan}/{@code --chan2}:
 * 1-based, with {@code 0} meaning grayscale. {@code cp4.py} instead slices the channel axis with
 * plain 0-based indices. These tests pin the translation between the two, and the bounds checks
 * that turn a mis-set script into a readable message instead of a Python IndexError.
 */
public class ChannelMappingTest {

    @Test
    void unspecifiedChannelsUseTheWholeTile() {
        // Every shipped example script leaves cellposeChannels() unset; Cellpose-SAM should then
        // see all exported channels, exactly as it does through the subprocess backend.
        Assertions.assertArrayEquals(new int[] {0}, ApposeBackend.resolveCp4Channels(0, null, 1));
        Assertions.assertArrayEquals(new int[] {0, 1}, ApposeBackend.resolveCp4Channels(0, null, 2));
        Assertions.assertArrayEquals(new int[] {0, 1, 2}, ApposeBackend.resolveCp4Channels(null, null, 3));
    }

    @Test
    void tileWiderThanCellposeSamIsTruncatedNotRejected() {
        Assertions.assertArrayEquals(new int[] {0, 1, 2}, ApposeBackend.resolveCp4Channels(null, null, 7));
    }

    @Test
    void cellposeChannelNumbersAreConvertedToZeroBasedIndices() {
        // cellposeChannels(1, 2) means "first and second exported channel" -> indices 0 and 1.
        Assertions.assertArrayEquals(new int[] {0, 1}, ApposeBackend.resolveCp4Channels(1, 2, 2));
        // Order is preserved: cellposeChannels(2, 1) is cytoplasm-then-nucleus.
        Assertions.assertArrayEquals(new int[] {1, 0}, ApposeBackend.resolveCp4Channels(2, 1, 2));
        // A grayscale 0 in either slot selects nothing on its own.
        Assertions.assertArrayEquals(new int[] {2}, ApposeBackend.resolveCp4Channels(3, 0, 3));
    }

    @Test
    void duplicateChannelRequestsAreCollapsed() {
        Assertions.assertArrayEquals(new int[] {0}, ApposeBackend.resolveCp4Channels(1, 1, 2));
    }

    @Test
    void outOfRangeChannelIsReportedAgainstTheExportedTile() {
        // The pre-fix bug: cellposeChannels(1, 2) sent as raw indices ran off the end of a
        // two-channel tile inside Python. It must now fail in Java with an actionable message.
        IllegalArgumentException e = Assertions.assertThrows(IllegalArgumentException.class,
                () -> ApposeBackend.resolveCp4Channels(2, 3, 2));
        Assertions.assertTrue(e.getMessage().contains("only has 2 channel(s)"), e.getMessage());
        Assertions.assertTrue(e.getMessage().contains("--chan"), e.getMessage());
    }

    @Test
    void cellpose3SingleChannelTileIsSentUntouched() {
        ApposeBackend.Cp3Layout layout = ApposeBackend.Cp3Layout.resolve(0, null, 1);
        Assertions.assertArrayEquals(new int[] {0}, layout.bands());
        Assertions.assertEquals(1, layout.packedChannels());
        Assertions.assertEquals(0, layout.cellChannel());
        Assertions.assertNull(layout.nucleiChannel());
    }

    @Test
    void cellpose3GrayscaleOverManyChannelsIsAveragedInJava() {
        // Above three channels Cellpose reads the channel axis as a Z axis and silently returns an
        // empty mask, so the grayscale average that its [0, 0] spec implies is done here instead.
        ApposeBackend.Cp3Layout layout = ApposeBackend.Cp3Layout.resolve(0, null, 7);
        Assertions.assertNull(layout.bands(), "null bands means 'average every channel'");
        Assertions.assertEquals(1, layout.packedChannels());
        Assertions.assertEquals(0, layout.cellChannel());
        Assertions.assertNull(layout.nucleiChannel());
    }

    @Test
    void cellpose3PacksTheRequestedChannelsAndRenumbersTheSpec() {
        // cellposeChannels(2, 1) on a 7-channel tile: send channels 1 and 0 of the tile, in that
        // order, and describe the packed pair as [1, 2].
        ApposeBackend.Cp3Layout layout = ApposeBackend.Cp3Layout.resolve(2, 1, 7);
        Assertions.assertArrayEquals(new int[] {1, 0}, layout.bands());
        Assertions.assertEquals(2, layout.packedChannels());
        Assertions.assertEquals(1, layout.cellChannel());
        Assertions.assertEquals(2, layout.nucleiChannel());
    }

    @Test
    void cellpose3SingleRequestedChannelBecomesAGrayscalePlane() {
        ApposeBackend.Cp3Layout layout = ApposeBackend.Cp3Layout.resolve(3, 0, 4);
        Assertions.assertArrayEquals(new int[] {2}, layout.bands());
        Assertions.assertEquals(0, layout.cellChannel());
        Assertions.assertNull(layout.nucleiChannel());
    }

    @Test
    void cellpose3OutOfRangeChannelIsReportedAgainstTheExportedTile() {
        IllegalArgumentException e = Assertions.assertThrows(IllegalArgumentException.class,
                () -> ApposeBackend.Cp3Layout.resolve(1, 3, 2));
        Assertions.assertTrue(e.getMessage().contains("channel 3"), e.getMessage());
        Assertions.assertTrue(e.getMessage().contains("only has 2 channel(s)"), e.getMessage());
    }
}
