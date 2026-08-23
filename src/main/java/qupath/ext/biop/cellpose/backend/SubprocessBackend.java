/*-
 * Copyright 2020-2021 BioImaging & Optics Platform BIOP, Ecole Polytechnique Fédérale de Lausanne
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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.biop.cellpose.TileFile;
import qupath.ext.biop.cmd.VirtualEnvironmentRunner;

import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * The default Cellpose backend: runs Cellpose (or Omnipose) as an external Python process through a
 * {@link VirtualEnvironmentRunner}, exchanging images through temporary TIFF files.
 * <p>
 * This class holds the Cellpose-detection run logic that used to live in
 * {@code Cellpose2D.runCellpose(...)} and {@code Cellpose2D.processCellposeFiles(...)}. It was
 * relocated here verbatim as part of introducing the pluggable {@link CellposeBackend} seam; the
 * command-line arguments, temporary-file naming and file processing are unchanged so that the
 * default transport behaves byte-for-byte as before. The one adaptation is that the mask reader is
 * invoked through the {@code tileReader} callback supplied by {@code Cellpose2D} (which delegates to
 * the unchanged {@code readObjectsFromTileFile}), so the reader itself stays in {@code Cellpose2D}.
 * The {@link VirtualEnvironmentRunner} factory also stays in {@code Cellpose2D} (training and QC
 * share it) and is provided here as a {@link Supplier}.
 */
public class SubprocessBackend implements CellposeBackend {

    private static final Logger logger = LoggerFactory.getLogger(SubprocessBackend.class);

    private final java.io.File tempDirectory;
    private final LinkedHashMap<String, String> parameters;
    private final String model;
    private final boolean disableGPU;
    private final boolean doReadResultsAsynchronously;
    private final Supplier<VirtualEnvironmentRunner> veRunnerSupplier;
    private final Consumer<TileFile> tileReader;

    /**
     * Create a subprocess backend.
     *
     * @param tempDirectory               directory holding the input tiles (and where masks are written)
     * @param parameters                  the raw Cellpose command-line flag map
     * @param model                       the pretrained model name or path
     * @param disableGPU                  whether to disable GPU usage
     * @param doReadResultsAsynchronously whether to read result files as they appear (experimental)
     * @param veRunnerSupplier            supplies a configured {@link VirtualEnvironmentRunner}
     * @param tileReader                  callback that reads detections from a tile's mask file (may be
     *                                    {@code null} when running Cellpose without reading results,
     *                                    e.g. training validation)
     */
    public SubprocessBackend(java.io.File tempDirectory,
                             LinkedHashMap<String, String> parameters,
                             String model,
                             boolean disableGPU,
                             boolean doReadResultsAsynchronously,
                             Supplier<VirtualEnvironmentRunner> veRunnerSupplier,
                             Consumer<TileFile> tileReader) {
        this.tempDirectory = tempDirectory;
        this.parameters = parameters;
        this.model = model;
        this.disableGPU = disableGPU;
        this.doReadResultsAsynchronously = doReadResultsAsynchronously;
        this.veRunnerSupplier = veRunnerSupplier;
        this.tileReader = tileReader;
    }

    /**
     * {@inheritDoc}
     * <p>
     * The {@link CellposeSegmentationParams} argument is ignored: the external Cellpose command is
     * built from the raw flag map supplied at construction, to keep the command byte-for-byte
     * identical to the historical behaviour.
     */
    @Override
    public void run(List<TileFile> allTiles, CellposeSegmentationParams params) throws InterruptedException, IOException {

        // Need to define the name of the command we are running. We used to be able to use 'cellpose' for both but not since Cellpose v2
        String runCommand = this.parameters.containsKey("omni") ? "omnipose" : "cellpose";
        VirtualEnvironmentRunner veRunner = veRunnerSupplier.get();

        // This is the list of commands after the 'python' call
        // We want to ignore all warnings to make sure the log is clean (-W ignore)
        // We want to be able to call the module by name (-m)
        // We want to make sure UTF8 mode is by default (-X utf8)
        List<String> cellposeArguments = new ArrayList<>(Arrays.asList("-Xutf8", "-W", "ignore", "-m", runCommand));

        cellposeArguments.add("--dir");
        cellposeArguments.add("" + this.tempDirectory);

        cellposeArguments.add("--pretrained_model");
        cellposeArguments.add(this.model);

        this.parameters.forEach((parameter, value) -> {
            cellposeArguments.add("--" + parameter);
            if (value != null) {
                cellposeArguments.add(value);
            }
        });

        // These all work for cellpose v2
        cellposeArguments.add("--save_tif");

        cellposeArguments.add("--no_npy");

        if (!this.disableGPU) cellposeArguments.add("--use_gpu");

        cellposeArguments.add("--verbose");

        veRunner.setArguments(cellposeArguments);

        // Finally, we can run Cellpose
        veRunner.runCommand(false);

        processCellposeFiles(veRunner, allTiles);
    }

    private void processCellposeFiles(VirtualEnvironmentRunner veRunner, List<TileFile> allTiles) throws CancellationException, InterruptedException, IOException {

        // Make sure that allTiles is not null, if it is, just return null
        // as we are likely just running validation and thus do not need to give any results back
        if (allTiles == null) {
            veRunner.getProcess().waitFor();
            return;
        }

        // Build a thread pool to process reading the images in parallel
        ExecutorService executor = Executors.newFixedThreadPool(5);

        if (!this.doReadResultsAsynchronously) {
            // We need to wait for the process to finish
            veRunner.getProcess().waitFor();
            allTiles.forEach(entry -> {
                executor.execute(() -> {
                    // Read the objects from the file
                    tileReader.accept(entry);
                });

            });
        } else { // Experimental file listening and running

            //Make a map of the original names and the expected names
            LinkedHashMap<java.io.File, TileFile> remainingFiles = allTiles.stream().map(entry -> {
                java.io.File expectedFile = entry.getLabelFile();
                return new AbstractMap.SimpleEntry<>(expectedFile, entry);
            }).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (a, b) -> b, LinkedHashMap::new));

            try {
                // We need to listen for changes in the temp folder
                veRunner.startWatchService(this.tempDirectory.toPath());

                // The command above will run in a separate thread, now we can start listening for the files changing
                while (!remainingFiles.isEmpty() && veRunner.getProcess().isAlive()) {
                    if (!veRunner.getProcess().isAlive()) {
                        // It's no longer running so check the exit code
                        int exitValue = veRunner.getProcess().exitValue();
                        if (exitValue != 0) {
                            throw new IOException("Cellpose process exited with value " + exitValue + ". Please check output above for indications of the problem.\nWill attempt to continue");
                        }
                    }

                    // Get the files that have changes
                    List<String> changedFiles = veRunner.getChangedFiles();

                    if (changedFiles.isEmpty()) {
                        continue;
                    }

                    // Find the tiles that corresponds to the changed files
                    LinkedHashMap<java.io.File, TileFile> finishedFiles = remainingFiles.entrySet().stream().filter(set -> {
                        // Create a file that matches the mask name
                        return changedFiles.contains(set.getKey().getName());
                    }).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (a, b) -> b, LinkedHashMap::new));

                    // Announce that these files are done
                    finishedFiles.forEach((key, tile) -> executor.execute(() -> {
                        // Read the objects from the file
                        tileReader.accept(tile);
                    }));

                    // Remove from the queue
                    finishedFiles.forEach((k, v) -> {
                        remainingFiles.remove(k);
                    });
                }
            } catch (IOException e) {
                logger.error(e.getMessage(), e);

            } finally {
                // No matter what, try and check if there are tiles left

                // Get the files that have changes
                List<String> changedFiles = veRunner.getChangedFiles();

                // Find the tiles that corresponds to the changed files
                LinkedHashMap<java.io.File, TileFile> finishedFiles = remainingFiles.entrySet().stream().filter(set -> {
                    // Create a file that matches the mask name
                    return changedFiles.contains(set.getKey().getName());
                }).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (a, b) -> b, LinkedHashMap::new));

                // Announce that these files are done
                finishedFiles.forEach((key, tile) -> {
                    executor.execute(() -> {
                        // Read the objects from the file
                        tileReader.accept(tile);
                    });
                });
                // Remove them from the list of remaining files

                veRunner.closeWatchService();

            }
        }

        executor.shutdown();
        executor.awaitTermination(10, TimeUnit.MINUTES);
    }

    /** No resources to release for the subprocess transport. */
    @Override
    public void close() {
        // no-op
    }
}
