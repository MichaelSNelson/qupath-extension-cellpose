/*-
 * Copyright 2026 BioImaging & Optics Platform (BIOP), Ecole Polytechnique Federale de Lausanne
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

/**
 * The transport used to run Cellpose from within QuPath.
 * <p>
 * This is Apache-2.0 original work for this fork.
 */
public enum CellposeTransport {

    /**
     * Run Cellpose as an external Python process, exchanging images through temporary TIFF files.
     * This is the default and historical behaviour of the extension.
     */
    SUBPROCESS,

    /**
     * Run Cellpose in-process through Appose, exchanging images through shared memory.
     * This is an opt-in alternative; the environment build/service lifecycle is adapted from the
     * BSD-3-Clause imglib2-cellpose project (see NOTICE).
     */
    APPOSE
}
