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

/**
 * The compute device the in-process {@link ApposeBackend} should run Cellpose on.
 * <p>
 * This is Apache-2.0 original work for this fork. The resolved device controls both which pixi
 * sub-environment is activated (a CUDA-enabled {@code cuNNN} variant vs the {@code cpu} variant)
 * and the {@code use_gpu} flag handed to the Cellpose scripts.
 */
public enum CellposeDevice {

    /**
     * Detect a usable GPU automatically (the default). Uses {@link ApposeEnvironments} GPU
     * detection; falls back to CPU when no NVIDIA GPU is found (and always on macOS, which has no
     * CUDA).
     */
    AUTO,

    /**
     * Force GPU: activate the CUDA sub-environment and request GPU execution regardless of
     * detection. Cellpose itself still falls back to CPU at runtime if no usable device is present.
     */
    GPU,

    /**
     * Force CPU: activate the CPU sub-environment and request CPU execution.
     */
    CPU
}
