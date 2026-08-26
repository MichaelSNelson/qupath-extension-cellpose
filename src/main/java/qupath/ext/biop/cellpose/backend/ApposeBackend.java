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
import qupath.lib.common.ThreadTools;
import qupath.ext.biop.cellpose.TileFile;
import qupath.ext.biop.cellpose.ui.PythonConsoleWindow;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
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
 * This is Apache-2.0 original work. Nothing touches the disk: for each tile the backend obtains the
 * pixels from {@code Cellpose2D} on demand, converts them to an Appose {@link NDArray} in shared
 * memory (through {@link NDArrays}), submits the vendored Cellpose script for the requested model
 * family, reads the label {@link NDArray} back, and hands it to the tile reader in memory. (A tile
 * that does have a file behind it - which only happens if one is constructed the subprocess way - is
 * still read from and written to disk, so the two paths can coexist.)
 * <p>
 * Tiles are processed by a bounded pool sized from QuPath's parallelism setting, so at most that
 * many tiles hold pixels at once and memory does not scale with the size of the region. The Cellpose
 * call itself is serialized: Appose multiplexes all tasks over one worker process and the vendored
 * scripts cache the model in a single Python global with no lock, so concurrent {@code model.eval}
 * calls would race on the model and on GPU memory. Extracting a tile and tracing its masks - the
 * CPU-bound work - happen outside that lock, so they overlap with another tile's segmentation.
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

    /** Attempts for the transient Appose worker "thread death" (see {@link #submit}). */
    private static final int THREAD_DEATH_ATTEMPTS = 3;
    private static final long THREAD_DEATH_RETRY_DELAY_MS = 500L;

    /** Cellpose-SAM (Cellpose 4) accepts at most three channels. */
    private static final int CELLPOSE_SAM_MAX_CHANNELS = 3;

    /**
     * Live Appose services, so a single JVM shutdown hook can close them if QuPath is force-quit
     * before {@link #close()} runs, preventing an orphaned Python subprocess. The Python-side
     * parent-watcher (injected into the init script) is the second line of defence.
     */
    private static final Set<Service> LIVE_SERVICES = ConcurrentHashMap.newKeySet();
    private static final AtomicBoolean SHUTDOWN_HOOK_INSTALLED = new AtomicBoolean(false);

    private final Consumer<TileFile> tileReader;

    /** Requested worker count; <= 0 means follow QuPath's own parallelism setting. */
    private final int requestedThreads;

    /**
     * Serializes the Python side. Appose multiplexes every task over one worker process, and the
     * vendored Cellpose scripts cache the model in a single Python global with no lock of their own,
     * so two concurrent {@code model.eval} calls would race on that model and on GPU memory. The
     * lock lives on the {@link Worker}, not on this backend, because the worker is shared between
     * runs. Tile extraction and mask tracing - the CPU-bound work worth parallelizing - stay outside
     * it.
     */
    private Object pythonLock() {
        return worker.lock;
    }

    private Worker worker;
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
        this(tileReader, -1);
    }

    /**
     * @param tileReader callback turning a finished tile's masks into candidate objects
     * @param nThreads   worker count for tile extraction and mask tracing; <= 0 follows
     *                   QuPath's parallelism setting
     */
    public ApposeBackend(Consumer<TileFile> tileReader, int nThreads) {
        this.tileReader = tileReader;
        installShutdownHook();
        this.requestedThreads = nThreads;
    }

    @Override
    public void run(List<TileFile> tiles, CellposeSegmentationParams params) throws IOException, InterruptedException {
        if (tiles == null || tiles.isEmpty()) {
            // Nothing to segment; the Appose backend is not used for the training/validation
            // "run cellpose only" case.
            return;
        }

        ensureService(params);
        initializeModel(params);

        int workers = Math.max(1, Math.min(
                requestedThreads > 0 ? requestedThreads : ThreadTools.getParallelism(),
                tiles.size()));
        logger.info("Segmenting {} tile(s) with {} worker(s)", tiles.size(), workers);

        // Bounded pool: at most `workers` tiles hold pixels at once, so memory scales with the
        // thread count rather than with the number of tiles. Each worker extracts its tile, waits
        // its turn for the (serialized) Cellpose call, then traces the masks while another worker
        // is in Python -- which is the overlap that makes this worth doing.
        ExecutorService pool = Executors.newFixedThreadPool(workers,
                ThreadTools.createThreadFactory("cellpose-appose-", true));
        List<Future<?>> futures = new ArrayList<>(tiles.size());
        try {
            for (TileFile tile : tiles)
                futures.add(pool.submit(() -> { processTile(tile, params); return null; }));

            for (Future<?> future : futures) {
                try {
                    future.get();
                } catch (ExecutionException e) {
                    // Stop the run on the first real failure, and don't let the remaining tiles
                    // keep the user waiting for an outcome that is already lost.
                    futures.forEach(f -> f.cancel(true));
                    Throwable cause = e.getCause();
                    if (cause instanceof IOException)
                        throw (IOException) cause;
                    if (cause instanceof InterruptedException)
                        throw (InterruptedException) cause;
                    throw new IOException("Cellpose segmentation failed: " + cause.getMessage(), cause);
                } catch (CancellationException e) {
                    throw new InterruptedException("Cellpose segmentation cancelled");
                }
            }
        } catch (InterruptedException e) {
            futures.forEach(f -> f.cancel(true));
            Thread.currentThread().interrupt();
            throw e;
        } finally {
            pool.shutdownNow();
        }
    }

    /**
     * Load the model once, before any tile is dispatched, so the workers never race to initialize
     * it. The vendored init script reads only the model-selection globals; the model it builds is
     * then cached in Python and reused by every subsequent task.
     */
    private void initializeModel(CellposeSegmentationParams params) throws IOException, InterruptedException {
        String signature = (params.isCustomModel() ? "custom:" : "named:") + params.getModel();
        synchronized (pythonLock()) {
            if (signature.equals(worker.loadedModel)) {
                logger.info("Reusing the Cellpose model already loaded in the Python worker ({})", params.getModel());
                return;
            }
            Map<String, Object> inputs = new HashMap<>();
            inputs.put("use_gpu", useGpu);
            inputs.put("custom_model", params.isCustomModel() ? params.getModel() : null);
            inputs.put("model_name", params.isCustomModel() ? null : params.getModel());
            // Clear first: if loading fails, the worker must not claim to hold a model it does not.
            worker.loadedModel = null;
            submit(initScript, inputs, "Initializing Cellpose model");
            worker.loadedModel = signature;
        }
    }

    /** Extract one tile, segment it, and hand the labels to the tile reader. */
    private void processTile(TileFile tile, CellposeSegmentationParams params) throws IOException, InterruptedException {
        // Cooperative cancellation: Cellpose's model.eval is a single blocking call that cannot be
        // interrupted mid-tile, so honor QuPath's Stop between tiles.
        if (Thread.currentThread().isInterrupted())
            throw new InterruptedException("Cellpose segmentation cancelled before tile "
                    + tile.getImageFile().getName());

        ImagePlus imp = tile.openImage();
        if (imp == null) {
            logger.warn("Could not obtain tile image {}; skipping", tile.getImageFile());
            return;
        }
        int width = imp.getWidth();
        int height = imp.getHeight();

        // Cellpose 3 gets a compacted array holding only the channels it will use (see Cp3Layout);
        // Cellpose 4 gets the whole stack, because cp4.py does its own slicing.
        Cp3Layout layout = params.isCellposeSam()
                ? null
                : Cp3Layout.resolve(params.getChannel1(), params.getChannel2(), imp.getStackSize());

        NDArray input = allocate(() -> layout == null ? NDArrays.fromImagePlus(imp) : layout.pack(imp));
        NDArray labels = allocate(() -> NDArrays.allocateLabels(width, height));
        ShortProcessor labelProcessor;
        try {
            Map<String, Object> inputs = buildInputs(params, imp.getStackSize(), layout, input, labels);
            synchronized (pythonLock()) {
                submit(runScript, inputs, "Segmenting tile " + tile.getImageFile().getName());
            }
            labelProcessor = NDArrays.labelsToShortProcessor(labels, width, height);
        } finally {
            closeQuietly(input);
            closeQuietly(labels);
            imp.close();
        }

        if (tile.isInMemory()) {
            // No mask file: hand the labels straight to the reader.
            tile.setLabels(labelProcessor);
        } else {
            ImagePlus maskImp = new ImagePlus(tile.getLabelFile().getName(), labelProcessor);
            IJ.save(maskImp, tile.getLabelFile().getAbsolutePath());
            maskImp.close();
        }

        // Turn the masks into candidate detections. Deliberately outside the Python lock, so this
        // runs while another worker is segmenting.
        tileReader.accept(tile);
    }

    /**
     * A Python worker kept alive between runs, together with the lock that serializes access to it
     * and a note of which model it currently holds loaded.
     * <p>
     * Starting a worker costs several seconds -- almost all of it importing torch and cellpose -- so
     * creating one per {@code detectObjects} call made every run after the first pay for a Python
     * interpreter it did not need. A two-stage script paid it twice; a project batch paid it once
     * per image. The worker therefore outlives the backend and is closed at JVM shutdown, or on
     * demand from {@code Extensions > Cellpose}, which is also how a user reclaims GPU memory.
     */
    private static final class Worker {
        final Service service;
        /** Serializes the Cellpose call; see {@link ApposeBackend#pythonLock}. */
        final Object lock = new Object();
        /** Model currently loaded in this worker, so an unchanged model is not reloaded. */
        String loadedModel;

        Worker(Service service) {
            this.service = service;
        }
    }

    /** Live workers, keyed by pixi sub-environment (model family + device). */
    private static final Map<String, Worker> WORKERS = new HashMap<>();

    /**
     * Shut down every cached Python worker, releasing the GPU memory they hold. Safe to call at any
     * time; the next run simply starts a fresh worker.
     */
    public static synchronized void shutdownWorkers() {
        if (WORKERS.isEmpty()) {
            logger.info("No Cellpose Python worker is running");
            return;
        }
        logger.info("Shutting down {} Cellpose Python worker(s)", WORKERS.size());
        WORKERS.values().forEach(w -> {
            try {
                ApposeEnvironments.withExtensionClassLoader(() -> {
                    w.service.close();
                    return null;
                });
            } catch (Exception e) {
                logger.warn("Error closing Appose Cellpose service: {}", e.getMessage(), e);
            } finally {
                LIVE_SERVICES.remove(w.service);
            }
        });
        WORKERS.clear();
    }

    private synchronized void ensureService(CellposeSegmentationParams params) throws IOException {
        if (worker != null)
            return;

        this.useGpu = ApposeEnvironments.resolveUseGpu(params.getDevice());
        String envName = ApposeEnvironments.envName(params.isCellposeSam(), useGpu);
        String scriptName = params.isCellposeSam() ? "cp4.py" : "cp3.py";
        String initName = params.isCellposeSam() ? "cp4_init.py" : "cp3_init.py";

        this.runScript = ApposeEnvironments.readResource(scriptName);
        this.initScript = ApposeEnvironments.readResource(initName);

        this.worker = acquireWorker(envName, params);
    }

    /** Reuse the worker for this environment if one is already running, otherwise start one. */
    private static synchronized Worker acquireWorker(String envName, CellposeSegmentationParams params) throws IOException {
        Worker existing = WORKERS.get(envName);
        if (existing != null) {
            // A cached worker is only useful if its process is still there. It may not be: the user
            // can shut it down from the menu, and a Python worker can die on its own. Replace a dead
            // one rather than handing it out and failing the run.
            if (existing.service.isAlive()) {
                logger.info("Reusing the running Cellpose Python worker for environment {}", envName);
                return existing;
            }
            logger.info("The cached Cellpose Python worker for environment {} is no longer running; starting a new one", envName);
            LIVE_SERVICES.remove(existing.service);
            WORKERS.remove(envName);
        }

        String cpUtils = ApposeEnvironments.readResource("cp_utils.py");
        // FIX: pre-import numpy FIRST (a numpy import after the stdin reader starts deadlocks on
        // Windows), then the parent-watcher, then cp_utils. init() replaces (not appends), so this
        // is one combined string.
        String init = "import numpy\n" + parentWatcherSnippet() + cpUtils;

        logger.info("Starting Appose Cellpose service (device={} -> environment {}, use_gpu={})",
                params.getDevice(), envName, ApposeEnvironments.resolveUseGpu(params.getDevice()));
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
            LIVE_SERVICES.add(created);
            Worker w = new Worker(created);
            WORKERS.put(envName, w);
            return w;
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
                                            Cp3Layout layout, NDArray input, NDArray labels) {
        Map<String, Object> inputs = new HashMap<>();

        // Images (shared memory).
        inputs.put("input", input);
        inputs.put("output_labels", labels);
        inputs.put("output_flows", null);

        // Axes: 2D tiles, so no Z or T. Channel axis is 0 when we send a (C, Y, X) stack -- which
        // for Cellpose 3 depends on how many channels the layout actually packed, not on how many
        // the tile has.
        inputs.put("t_axis", null);
        inputs.put("z_axis", null);
        int sentChannels = layout == null ? nChannels : layout.packedChannels();
        inputs.put("channel_axis", sentChannels > 1 ? Integer.valueOf(0) : null);

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
            // Cellpose 4 (cp4.py) channel handling. cp4.py slices the channel axis DIRECTLY
            // (input_image[..., channels, :, :]), so its chan0/chan1/chan2 are 0-based array
            // indices -- NOT cellpose's --chan/--chan2 convention, which is what the builder's
            // cellposeChannels(...) carries. Translate before sending; see resolveCp4Channels.
            inputs.put("n_channels", nChannels);
            int[] indices = resolveCp4Channels(params.getChannel1(), params.getChannel2(), nChannels);
            inputs.put("chan0", indices.length > 0 ? Integer.valueOf(indices[0]) : null);
            inputs.put("chan1", indices.length > 1 ? Integer.valueOf(indices[1]) : null);
            inputs.put("chan2", indices.length > 2 ? Integer.valueOf(indices[2]) : null);
        } else {
            // Cellpose 3 (cp3.py) channel handling: cp3.py passes these straight to model.eval as
            // `channels`, which IS the --chan/--chan2 convention. Because the layout has already
            // packed the tile down to just the channels in use, the spec refers to positions in
            // that packed array, not in the original tile.
            inputs.put("cell_channel", layout.cellChannel());
            inputs.put("nuclei_channel", layout.nucleiChannel());
        }

        return inputs;
    }

    /**
     * Translate cellpose's {@code --chan}/{@code --chan2} channel numbers into the 0-based channel
     * indices that {@code cp4.py} uses to slice the tile.
     * <p>
     * The two conventions differ and must not be conflated:
     * <ul>
     *     <li>{@code --chan}/{@code --chan2} (what {@code CellposeBuilder.cellposeChannels(a, b)}
     *     sets) are 1-based, with {@code 0} meaning "grayscale / not specified";</li>
     *     <li>{@code cp4.py} does {@code input_image[..., channels, :, :]}, so its values are plain
     *     0-based indices into the exported tile's channel axis.</li>
     * </ul>
     * Passing the former as the latter both shifts every channel by one and puts
     * {@code cellposeChannels(1, 2)} out of bounds on a two-channel tile.
     * <p>
     * When no channel is specified (the default, and what every shipped example script does), all
     * exported channels are used -- matching the subprocess backend, which hands Cellpose-SAM the
     * whole tile. Cellpose-SAM accepts at most three channels, so a wider tile is truncated with a
     * warning rather than failing.
     *
     * @param chan      the {@code --chan} value, or null if unset
     * @param chan2     the {@code --chan2} value, or null if unset
     * @param nChannels the number of channels in the exported tile
     * @return the 0-based channel indices to send as {@code chan0}/{@code chan1}/{@code chan2}
     * @throws IllegalArgumentException if a requested channel does not exist in the tile
     */
    static int[] resolveCp4Channels(Integer chan, Integer chan2, int nChannels) {
        List<Integer> requested = new ArrayList<>();
        for (Integer value : new Integer[] {chan, chan2}) {
            // 0 means "grayscale/unspecified" in the cellpose convention, so it selects nothing here.
            if (value == null || value == 0)
                continue;
            int index = value - 1;
            if (index >= nChannels)
                throw new IllegalArgumentException(channelOutOfRangeMessage(chan, chan2, value, nChannels));
            if (!requested.contains(index))
                requested.add(index);
        }

        if (requested.isEmpty()) {
            // Nothing specified: use the whole tile, as the subprocess backend does.
            int used = Math.min(nChannels, CELLPOSE_SAM_MAX_CHANNELS);
            if (used < nChannels)
                logger.warn("Cellpose-SAM accepts at most {} channels; using the first {} of the {} exported channels. "
                        + "Use .channels(...) to export fewer channels, or cellposeChannels(...) to choose explicitly.",
                        CELLPOSE_SAM_MAX_CHANNELS, used, nChannels);
            int[] all = new int[used];
            for (int i = 0; i < used; i++)
                all[i] = i;
            return all;
        }

        int[] indices = new int[requested.size()];
        for (int i = 0; i < indices.length; i++)
            indices[i] = requested.get(i);
        return indices;
    }

    /**
     * How a tile's channels are packed for Cellpose 3, together with the cellpose channel spec that
     * describes the packed array.
     * <p>
     * Cellpose is handed only the channels it will actually use. Sending the whole stack and letting
     * Cellpose pick fails above three channels: it mistakes the channel axis for a Z axis -- logging
     * {@code "z_axis not specified, assuming it is dim 0"} -- and returns an empty mask with no
     * error, even when {@code channel_axis} is passed. Packing here also means the spec is a fixed
     * {@code [0, 0]} or {@code [1, 2]} regardless of which channels of the tile were requested.
     * <p>
     * The grayscale case is averaged rather than truncated, because that is what Cellpose itself
     * does for the {@code [0, 0]} spec, so results are unchanged for tiles it already handled.
     *
     * @param bands         0-based slice indices to send, in order, or null to average all channels
     * @param cellChannel   the cellpose {@code --chan} value describing the packed array
     * @param nucleiChannel the cellpose {@code --chan2} value describing the packed array, or null
     */
    record Cp3Layout(int[] bands, Integer cellChannel, Integer nucleiChannel) {

        /** Number of channels actually sent to Python (1 means a plain 2D plane). */
        int packedChannels() {
            return bands == null ? 1 : bands.length;
        }

        /** Build the input NDArray for this layout. The caller must close it. */
        NDArray pack(ImagePlus imp) {
            return bands == null ? NDArrays.meanOfChannels(imp) : NDArrays.fromImagePlusChannels(imp, bands);
        }

        /**
         * Work out the packing for a tile from the builder's cellpose channel numbers.
         *
         * @param chan      the {@code --chan} value, or null if unset
         * @param chan2     the {@code --chan2} value, or null if unset
         * @param nChannels the number of channels in the exported tile
         * @return the layout to use
         * @throws IllegalArgumentException if a requested channel does not exist in the tile
         */
        static Cp3Layout resolve(Integer chan, Integer chan2, int nChannels) {
            List<Integer> requested = new ArrayList<>();
            for (Integer value : new Integer[] {chan, chan2}) {
                // 0 means "grayscale/unspecified" in the cellpose convention, so it selects nothing.
                if (value == null || value == 0)
                    continue;
                if (value > nChannels)
                    throw new IllegalArgumentException(channelOutOfRangeMessage(chan, chan2, value, nChannels));
                if (!requested.contains(value - 1))
                    requested.add(value - 1);
            }

            if (requested.isEmpty()) {
                // Grayscale over the whole tile. A single-channel tile needs no averaging, so send
                // it untouched and keep its original pixel type.
                return nChannels == 1
                        ? new Cp3Layout(new int[] {0}, 0, null)
                        : new Cp3Layout(null, 0, null);
            }
            if (requested.size() == 1)
                return new Cp3Layout(new int[] {requested.get(0)}, 0, null);
            return new Cp3Layout(new int[] {requested.get(0), requested.get(1)}, 1, 2);
        }
    }

    private static String channelOutOfRangeMessage(Integer chan, Integer chan2, int value, int nChannels) {
        return String.format(
                "cellposeChannels(%s, %s) asks for channel %d, but the exported tile only has %d channel(s). "
                        + "These values are cellpose's --chan/--chan2 (1-based, 0 = grayscale) and must refer to "
                        + "channels selected by .channels(...) in the builder.",
                chan, chan2, value, nChannels);
    }

    /**
     * Submit one script to the worker, retrying the documented Appose "thread death" flake.
     * <p>
     * Appose occasionally reports {@code Task failed: thread death} when a worker task thread dies
     * before it reports completion, most often on the first task after a service starts. It is
     * transient and a plain re-submit succeeds. Our tasks are idempotent -- model init just rebuilds
     * the model, and a segmentation task overwrites the shared-memory label buffer -- so retrying is
     * safe. Only this specific failure is retried; a genuine Python error is surfaced immediately.
     */
    private void submit(String script, Map<String, Object> inputs, String description) throws IOException, InterruptedException {
        IOException last = null;
        for (int attempt = 1; attempt <= THREAD_DEATH_ATTEMPTS; attempt++) {
            try {
                submitOnce(script, inputs, description);
                return;
            } catch (IOException e) {
                if (!isThreadDeath(e))
                    throw e;
                last = e;
                if (attempt < THREAD_DEATH_ATTEMPTS) {
                    logger.warn("{}: Appose reported worker thread death (attempt {} of {}); retrying",
                            description, attempt, THREAD_DEATH_ATTEMPTS);
                    Thread.sleep(THREAD_DEATH_RETRY_DELAY_MS * attempt);
                }
            }
        }
        throw new IOException(description + " failed after " + THREAD_DEATH_ATTEMPTS
                + " attempts: Appose worker thread died each time", last);
    }

    /** True if this failure is the transient Appose worker "thread death", not a Python error. */
    private static boolean isThreadDeath(Throwable t) {
        for (Throwable cause = t; cause != null; cause = cause.getCause()) {
            String message = cause.getMessage();
            if (message != null && message.toLowerCase().contains("thread death"))
                return true;
        }
        return false;
    }

    private void submitOnce(String script, Map<String, Object> inputs, String description) throws IOException, InterruptedException {
        Task task;
        try {
            task = ApposeEnvironments.withExtensionClassLoader(() -> {
                Task t = worker.service.task(script, inputs);
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
        // The Python worker deliberately outlives this backend, so the next run does not pay for a
        // fresh interpreter and model load. It is closed at JVM shutdown, or on demand through
        // Extensions > Cellpose. Only the reference is dropped here.
        worker = null;
    }
}
