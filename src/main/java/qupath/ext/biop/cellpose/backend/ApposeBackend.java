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

import ij.IJ;
import ij.ImagePlus;
import ij.process.ShortProcessor;
import org.apposed.appose.NDArray;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;
import org.apposed.appose.Service.TaskStatus;
import org.apposed.appose.TaskEvent;
import org.apposed.appose.TaskException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.biop.cellpose.TileFile;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * In-process Cellpose segmentation backend based on Appose.
 * <p>
 * This is Apache-2.0 original work. For each tile, this backend reads the input tile image that
 * {@code Cellpose2D} already saved to disk, converts it to an Appose {@link NDArray} in shared
 * memory (through {@link NDArrays}), submits the vendored Cellpose script for the requested model
 * family, reads the label {@link NDArray} back, and writes it to the tile's {@code _cp_masks.tif}
 * file exactly where the existing subprocess path would - so the downstream mask reader in
 * {@code Cellpose2D} stays untouched. After each tile's masks are written, the tile reader supplied
 * by {@code Cellpose2D} turns those masks into candidate detections.
 * <p>
 * The Appose service lifecycle (build the pixi environment, activate the model-family
 * sub-environment, initialize the worker with {@code cp_utils.py}, then load the model with
 * {@code cpX_init.py} and run {@code cpX.py} per tile) is adapted, with attribution, from the
 * BSD-3-Clause imglib2-cellpose project (see NOTICE). The input-global keys supplied to the scripts
 * are those the scripts themselves read (learned by reading {@code cp3.py}/{@code cp4.py}); the
 * BufferedImage/ImageProcessor to NDArray marshalling is written independently against the raw
 * Appose NDArray API and does not reuse the ImgLib2 RAI bridge.
 * <p>
 * The service is persistent for the lifetime of this backend (one detection run): the model is
 * loaded once and reused across all tiles. It is closed by {@link #close()}.
 */
public class ApposeBackend implements CellposeBackend {

    private static final Logger logger = LoggerFactory.getLogger(ApposeBackend.class);

    private final Consumer<TileFile> tileReader;

    private Service service;
    private String runScript;
    private String initScript;

    /**
     * Create an Appose backend.
     *
     * @param tileReader callback that reads detections from a tile's mask file once it exists;
     *                   {@code Cellpose2D} supplies one that delegates to its (unchanged) mask reader
     */
    public ApposeBackend(Consumer<TileFile> tileReader) {
        this.tileReader = tileReader;
    }

    @Override
    public void run(List<TileFile> tiles, CellposeSegmentationParams params) throws IOException, InterruptedException {
        if (tiles == null || tiles.isEmpty()) {
            // Nothing to segment; the Appose backend is not used for the training/validation
            // "run cellpose only" case.
            return;
        }

        ensureService(params);

        boolean modelInitialized = false;
        for (TileFile tile : tiles) {
            ImagePlus imp = IJ.openImage(tile.getImageFile().getAbsolutePath());
            if (imp == null) {
                logger.warn("Could not open tile image {}; skipping", tile.getImageFile());
                continue;
            }
            int width = imp.getWidth();
            int height = imp.getHeight();

            NDArray input = NDArrays.fromImagePlus(imp);
            NDArray labels = NDArrays.allocateLabels(width, height);
            try {
                Map<String, Object> inputs = buildInputs(params, imp.getStackSize(), input, labels);

                if (!modelInitialized) {
                    submit(initScript, inputs, "Initializing Cellpose model");
                    modelInitialized = true;
                }
                submit(runScript, inputs, "Segmenting tile " + tile.getImageFile().getName());

                ShortProcessor labelProcessor = NDArrays.labelsToShortProcessor(labels, width, height);
                ImagePlus maskImp = new ImagePlus(tile.getLabelFile().getName(), labelProcessor);
                IJ.save(maskImp, tile.getLabelFile().getAbsolutePath());
                maskImp.close();
            } finally {
                input.close();
                labels.close();
                imp.close();
            }

            // Read detections from the mask we just wrote, using Cellpose2D's unchanged reader.
            tileReader.accept(tile);
        }
    }

    private synchronized void ensureService(CellposeSegmentationParams params) throws IOException {
        if (service != null)
            return;
        String envName = ApposeEnvironments.envName(params.isCellposeSam());
        String scriptName = params.isCellposeSam() ? "cp4.py" : "cp3.py";
        String initName = params.isCellposeSam() ? "cp4_init.py" : "cp3_init.py";

        this.runScript = ApposeEnvironments.readResource(scriptName);
        this.initScript = ApposeEnvironments.readResource(initName);
        String cpUtils = ApposeEnvironments.readResource("cp_utils.py");

        logger.info("Starting Appose Cellpose service (environment {})", envName);
        try {
            this.service = ApposeEnvironments.getEnvironment().activate(envName).python().init(cpUtils);
        } catch (org.apposed.appose.BuildException e) {
            throw new IOException("Failed to activate the Appose environment '" + envName + "'", e);
        }
    }

    /**
     * Build the map of input globals for the Cellpose scripts. The keys are exactly those that
     * {@code cp3.py}/{@code cp4.py} (and their {@code *_init.py}) read from {@code globals()} or
     * reference by name. Values follow the scripts' expectations: numeric parameters, booleans, the
     * shared-memory NDArrays for {@code input}/{@code output_labels}, and {@code null} where the
     * scripts expect Python {@code None}.
     */
    private static Map<String, Object> buildInputs(CellposeSegmentationParams params, int nChannels,
                                                   NDArray input, NDArray labels) {
        Map<String, Object> inputs = new HashMap<>();

        // Images (shared memory).
        inputs.put("input", input);
        inputs.put("output_labels", labels);
        inputs.put("output_flows", null);

        // Axes: 2D tiles, so no Z or T. Channel axis is 0 when we send a (C, Y, X) stack.
        inputs.put("t_axis", null);
        inputs.put("z_axis", null);
        inputs.put("channel_axis", nChannels > 1 ? Integer.valueOf(0) : null);

        // Model selection.
        boolean custom = params.isCustomModel();
        inputs.put("custom_model", custom ? params.getModel() : null);
        inputs.put("model_name", custom ? null : params.getModel());

        // Core parameters.
        inputs.put("diameter", params.getDiameter());
        inputs.put("use_3D", params.isDo3D());
        inputs.put("flow_threshold", params.getFlowThreshold());
        inputs.put("cellprob_threshold", params.getCellprobThreshold());
        inputs.put("use_gpu", params.isUseGpu());

        // Fixed defaults for the 2D-tile case (mirroring the script defaults).
        inputs.put("stitch_threshold", 0.0);
        inputs.put("anisotropy", 1.0);
        inputs.put("compute_flows", false);
        inputs.put("resample", true);
        inputs.put("normalize", true);
        inputs.put("min_size", 15.0);
        inputs.put("tile_overlap", 0.1);
        inputs.put("flow3D_smooth", 0);
        inputs.put("niter", null);

        if (params.isCellposeSam()) {
            // Cellpose 4 (cp4.py) channel handling.
            inputs.put("n_channels", nChannels);
            inputs.put("chan0", params.getChannel1());
            inputs.put("chan1", params.getChannel2());
            inputs.put("chan2", null);
        } else {
            // Cellpose 3 (cp3.py) channel handling.
            inputs.put("cell_channel", params.getChannel1());
            inputs.put("nuclei_channel", params.getChannel2());
        }

        return inputs;
    }

    private void submit(String script, Map<String, Object> inputs, String description) throws IOException, InterruptedException {
        Task task = service.task(script, inputs);
        task.listen(ApposeBackend::relay);
        task.start();
        try {
            task.waitFor();
        } catch (TaskException e) {
            throw new IOException(description + " failed: " + e.getMessage(), e);
        }
        if (task.status != TaskStatus.COMPLETE) {
            throw new IOException(description + " failed with status " + task.status + ": " + task.error);
        }
    }

    private static void relay(TaskEvent event) {
        if (event.message != null && !event.message.isEmpty()) {
            if (event.maximum > 0)
                logger.info("Cellpose: {} ({}/{})", event.message, event.current, event.maximum);
            else
                logger.info("Cellpose: {}", event.message);
        }
    }

    @Override
    public void close() {
        if (service != null) {
            try {
                service.close();
            } catch (Exception e) {
                logger.warn("Error closing Appose Cellpose service: {}", e.getMessage(), e);
            } finally {
                service = null;
            }
        }
    }
}
