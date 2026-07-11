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

import qupath.ext.biop.cellpose.TileFile;

import java.io.IOException;
import java.util.List;

/**
 * A pluggable backend that runs Cellpose segmentation on a set of already-saved input tiles and
 * makes the resulting detections available on each tile.
 * <p>
 * Two implementations are provided:
 * <ul>
 *     <li>{@link SubprocessBackend} - the default, unchanged external-process transport that
 *     exchanges images through temporary TIFF files;</li>
 *     <li>{@link ApposeBackend} - an opt-in in-process transport based on Appose.</li>
 * </ul>
 * <p>
 * Whichever backend is used, the contract is the same: after {@link #run(List, CellposeSegmentationParams)}
 * returns, every tile in the list has had its candidate detections populated (through the tile
 * reader that the owning {@code Cellpose2D} supplies at construction time). Everything downstream
 * (overlap resolution, measurements, ...) is therefore independent of the chosen transport.
 * <p>
 * This is Apache-2.0 original work for this fork.
 */
public interface CellposeBackend extends AutoCloseable {

    /**
     * Run segmentation on the given tiles, populating each tile's candidate detections.
     *
     * @param tiles  the tiles to process; a {@code null} value is permitted for the
     *               "run-cellpose-only" case used by training/validation, in which no detections
     *               are read back
     * @param params the segmentation parameters (used by the Appose backend; the subprocess
     *               backend builds its command from the raw Cellpose flag map instead and ignores
     *               this argument)
     * @throws IOException          if reading/writing images or communicating with Python fails
     * @throws InterruptedException if the running thread is interrupted
     */
    void run(List<TileFile> tiles, CellposeSegmentationParams params) throws IOException, InterruptedException;

    /**
     * Release any resources held by this backend (e.g. a persistent Appose service).
     * The subprocess backend holds no resources and closing it is a no-op.
     */
    @Override
    void close();

    /**
     * Resolve which transport to use, given an optional builder-level override and the
     * extension-wide preference. A non-null builder flag always wins; otherwise the preference is
     * used; if both are null, the default {@link CellposeTransport#SUBPROCESS} is returned.
     *
     * @param builderTransport the transport set on the builder, or null if unset
     * @param preference       the extension-wide preference, or null
     * @return the resolved transport (never null)
     */
    static CellposeTransport resolveTransport(CellposeTransport builderTransport, CellposeTransport preference) {
        if (builderTransport != null)
            return builderTransport;
        if (preference != null)
            return preference;
        return CellposeTransport.SUBPROCESS;
    }
}
