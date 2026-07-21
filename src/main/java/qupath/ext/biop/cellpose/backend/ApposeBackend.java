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
import qupath.ext.biop.cellpose.ui.PythonConsoleWindow;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
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

    /**
     * Live Appose services, so a single JVM shutdown hook can close them if QuPath is force-quit
     * before {@link #close()} runs, preventing an orphaned Python subprocess. The Python-side
     * parent-watcher (injected into the init script) is the second line of defence.
     */
    private static final Set<Service> LIVE_SERVICES = ConcurrentHashMap.newKeySet();
    private static final AtomicBoolean SHUTDOWN_HOOK_INSTALLED = new AtomicBoolean(false);

    private final Consumer<TileFile> tileReader;

    private Service service;
    private String runScript;
    private String initScript;
    private boolean useGpu;

    /**
     * Create an Appose backend.
     *
     * @param tileReader callback that reads detections from a tile's mask file once it exists;
     *                   {@code Cellpose2D} supplies one that delegates to its (unchanged) mask reader
     */
    public ApposeBackend(Consumer<TileFile> tileReader) {
        this.tileReader = tileReader;
        installShutdownHook();
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
            // Cooperative cancellation: Cellpose's model.eval is a single blocking call that cannot
            // be interrupted mid-tile, so honor QuPath's Stop between tiles.
            if (Thread.currentThread().isInterrupted())
                throw new InterruptedException("Cellpose segmentation cancelled before tile "
                        + tile.getImageFile().getName());

            ImagePlus imp = IJ.openImage(tile.getImageFile().getAbsolutePath());
            if (imp == null) {
                logger.warn("Could not open tile image {}; skipping", tile.getImageFile());
                continue;
            }
            int width = imp.getWidth();
            int height = imp.getHeight();

            NDArray input = allocate(() -> NDArrays.fromImagePlus(imp));
            NDArray labels = allocate(() -> NDArrays.allocateLabels(width, height));
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
                closeQuietly(input);
                closeQuietly(labels);
                imp.close();
            }

            // Read detections from the mask we just wrote, using Cellpose2D's unchanged reader.
            tileReader.accept(tile);
        }
    }

    private synchronized void ensureService(CellposeSegmentationParams params) throws IOException {
        if (service != null)
            return;

        this.useGpu = ApposeEnvironments.resolveUseGpu(params.getDevice());
        String envName = ApposeEnvironments.envName(params.isCellposeSam(), useGpu);
        String scriptName = params.isCellposeSam() ? "cp4.py" : "cp3.py";
        String initName = params.isCellposeSam() ? "cp4_init.py" : "cp3_init.py";

        this.runScript = ApposeEnvironments.readResource(scriptName);
        this.initScript = ApposeEnvironments.readResource(initName);
        String cpUtils = ApposeEnvironments.readResource("cp_utils.py");

        // FIX: pre-import numpy FIRST (a numpy import after the stdin reader starts deadlocks on
        // Windows), then the parent-watcher, then cp_utils. init() replaces (not appends), so this
        // is one combined string.
        String init = "import numpy\n" + parentWatcherSnippet() + cpUtils;

        logger.info("Starting Appose Cellpose service (device={} -> environment {}, use_gpu={})",
                params.getDevice(), envName, useGpu);
        try {
            Service created = ApposeEnvironments.withExtensionClassLoader(() -> {
                Service svc = ApposeEnvironments.getEnvironment().activate(envName).python();
                // Route Python diagnostics to the log and the user-visible console (stdout is the
                // Appose IPC channel, so this is the only way users see tracebacks at runtime). The
                // debug channel carries the full IPC protocol -- including the entire task script,
                // license header and all, on every call -- which is dev-only noise. Keep the raw
                // stream in the log at debug level, but show the console only human-readable lines
                // (Python warnings/tracebacks and failures), not the routine request/response JSON.
                svc.debug(msg -> {
                    logger.debug("[Cellpose Python] {}", msg);
                    if (isConsoleWorthy(msg))
                        PythonConsoleWindow.appendMessage(msg);
                });
                svc.init(init);
                return svc;
            });
            this.service = created;
            LIVE_SERVICES.add(created);
        } catch (org.apposed.appose.BuildException e) {
            throw new IOException("Failed to activate the Appose environment '" + envName + "'", e);
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new IOException("Failed to start the Appose Cellpose service for environment '" + envName + "'", e);
        }
    }

    /**
     * Build the map of input globals for the Cellpose scripts. The keys are exactly those that
     * {@code cp3.py}/{@code cp4.py} (and their {@code *_init.py}) read from {@code globals()} or
     * reference by name. Values follow the scripts' expectations: numeric parameters, booleans, the
     * shared-memory NDArrays for {@code input}/{@code output_labels}, and {@code null} where the
     * scripts expect Python {@code None}.
     */
    private Map<String, Object> buildInputs(CellposeSegmentationParams params, int nChannels,
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
        // Use the resolved device, not the raw builder flag: this stays consistent with the pixi
        // sub-environment we activated (cpu vs cuNNN).
        inputs.put("use_gpu", useGpu);

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
        Task task;
        try {
            task = ApposeEnvironments.withExtensionClassLoader(() -> {
                Task t = service.task(script, inputs);
                t.listen(ApposeBackend::relay);
                t.start();
                return t;
            });
        } catch (Exception e) {
            throw new IOException(description + " failed to start: " + e.getMessage(), e);
        }

        try {
            ApposeEnvironments.withExtensionClassLoader(() -> {
                task.waitFor();
                return null;
            });
        } catch (InterruptedException e) {
            // QuPath asked to stop while this tile was in flight: request cancellation of the task
            // and propagate so run() halts cleanly. The service is still closed by close().
            cancelQuietly(task);
            Thread.currentThread().interrupt();
            throw e;
        } catch (TaskException e) {
            throw new IOException(description + " failed: " + e.getMessage(), e);
        } catch (Exception e) {
            // Any other failure from the classloader-wrapped waitFor (e.g. a runtime error in the
            // Appose plumbing): surface it as an IOException.
            throw new IOException(description + " failed: " + e.getMessage(), e);
        }

        if (task.status != TaskStatus.COMPLETE) {
            throw new IOException(description + " failed with status " + task.status + ": " + task.error);
        }
    }

    private void cancelQuietly(Task task) {
        try {
            ApposeEnvironments.withExtensionClassLoader(() -> {
                task.cancel();
                return null;
            });
        } catch (Exception e) {
            logger.debug("Error cancelling Cellpose task: {}", e.getMessage());
        }
    }

    /** Allocate an NDArray under the extension classloader (ShmFactory ServiceLoader needs it). */
    private static NDArray allocate(Callable<NDArray> allocation) throws IOException {
        try {
            return ApposeEnvironments.withExtensionClassLoader(allocation);
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new IOException("Failed to allocate a shared-memory array for Cellpose: " + e.getMessage(), e);
        }
    }

    private static void closeQuietly(NDArray ndArray) {
        if (ndArray == null)
            return;
        try {
            ApposeEnvironments.withExtensionClassLoader(() -> {
                ndArray.close();
                return null;
            });
        } catch (Exception e) {
            logger.warn("Error closing shared-memory array: {}", e.getMessage(), e);
        }
    }

    private static void relay(TaskEvent event) {
        if (event.message != null && !event.message.isEmpty()) {
            String line;
            if (event.maximum > 0)
                line = "Cellpose: " + event.message + " (" + event.current + "/" + event.maximum + ")";
            else
                line = "Cellpose: " + event.message;
            logger.info(line);
            PythonConsoleWindow.appendMessage(line);
        }
    }

    /**
     * Decide whether a raw Appose debug line belongs in the user-facing Python console. The debug
     * channel carries the whole IPC protocol: routine task request/response JSON that embeds the
     * entire task script (and its BSD-3 license header) on every call. That is dev-only noise, so we
     * drop it -- but we keep everything human-readable (Python {@code [WORKER-*]} warnings and
     * tracebacks, pixi output, plain text) and any protocol line that reports a failure/error, so
     * runtime problems still surface. Scripted progress arrives separately via {@link #relay}.
     */
    private static boolean isConsoleWorthy(String msg) {
        if (msg == null || msg.isBlank())
            return false;
        String t = msg.strip();
        boolean protocolJson = t.indexOf('{') >= 0
                && (t.contains("\"requestType\"") || t.contains("\"responseType\"") || t.contains("\"script\""));
        if (!protocolJson)
            return true;
        String lower = t.toLowerCase();
        return lower.contains("failure") || lower.contains("\"error\"")
                || lower.contains("traceback") || lower.contains("exception");
    }

    /**
     * A small Python daemon, injected into the init script, that watches the parent (QuPath) process
     * and exits this worker if the parent dies. This prevents an orphaned python.exe if QuPath is
     * force-quit before the JVM shutdown hook can close the service. On Windows it uses
     * {@code OpenProcess}, NOT {@code os.kill(pid, 0)} (signal 0 on Windows crashes the target).
     * Internal names use a leading underscore so the worker does not export them to task scripts.
     */
    private static String parentWatcherSnippet() {
        return String.join("\n",
                "import os as _os, sys as _sys, threading as _thr, time as _time",
                "def _watch_parent():",
                "    _ppid = _os.getppid()",
                "    _is_win = _sys.platform.startswith('win')",
                "    _k = None",
                "    if _is_win:",
                "        import ctypes as _ct",
                "        _k = _ct.windll.kernel32",
                "    while True:",
                "        _time.sleep(2.0)",
                "        _alive = True",
                "        try:",
                "            if _is_win:",
                "                _h = _k.OpenProcess(0x1000, False, _ppid)",
                "                if not _h:",
                "                    _alive = False",
                "                else:",
                "                    _code = _ct.c_ulong(0)",
                "                    if _k.GetExitCodeProcess(_h, _ct.byref(_code)) and _code.value != 259:",
                "                        _alive = False",
                "                    _k.CloseHandle(_h)",
                "            else:",
                "                _os.kill(_ppid, 0)",
                "        except Exception:",
                "            _alive = False",
                "        if not _alive:",
                "            _os._exit(1)",
                "_thr.Thread(target=_watch_parent, daemon=True).start()",
                "");
    }

    private static void installShutdownHook() {
        if (SHUTDOWN_HOOK_INSTALLED.compareAndSet(false, true)) {
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                for (Service svc : LIVE_SERVICES) {
                    try {
                        svc.close();
                    } catch (Exception e) {
                        try {
                            svc.kill();
                        } catch (Exception ignore) {
                            // best effort at JVM shutdown
                        }
                    }
                }
            }, "cellpose-appose-shutdown"));
        }
    }

    @Override
    public void close() {
        if (service != null) {
            Service closing = service;
            try {
                ApposeEnvironments.withExtensionClassLoader(() -> {
                    closing.close();
                    return null;
                });
            } catch (Exception e) {
                logger.warn("Error closing Appose Cellpose service: {}", e.getMessage(), e);
            } finally {
                LIVE_SERVICES.remove(closing);
                service = null;
            }
        }
    }
}
