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

import org.apposed.appose.Appose;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

/**
 * Builds and caches the Appose (pixi) environment used by the in-process {@link ApposeBackend}.
 * <p>
 * This is Apache-2.0 original work. The environment-build call and the Cellpose 3 vs Cellpose-SAM
 * (Cellpose 4) environment-name/CUDA-suffix scheme are adapted, with attribution, from the
 * BSD-3-Clause imglib2-cellpose project ({@code net.imglib2.cellpose.Cellpose} and
 * {@code CellposeRunner}; see NOTICE). The pixi manifest is a single vendored {@code pixi.toml}
 * declaring the {@code cp3-*}/{@code cp4-*} sub-environments; the two Cellpose major versions
 * cannot share one environment, so callers activate the sub-environment matching the model family.
 */
final class ApposeEnvironments {

    private static final Logger logger = LoggerFactory.getLogger(ApposeEnvironments.class);

    /** Root of the vendored Appose resources (scripts and pixi manifest). */
    static final String RESOURCE_ROOT = "/qupath/ext/biop/cellpose/appose/";

    private static final Object LOCK = new Object();
    private static volatile Environment environment;

    private ApposeEnvironments() {}

    /**
     * Build (once) and return the shared Appose environment from the vendored pixi manifest. The
     * first call triggers the pixi install, which can be slow; subsequent calls return the cached
     * environment.
     *
     * @return the shared Appose environment
     * @throws IOException if the pixi manifest cannot be read or the environment cannot be built
     */
    static Environment getEnvironment() throws IOException {
        Environment local = environment;
        if (local != null)
            return local;
        synchronized (LOCK) {
            if (environment == null) {
                String pixiToml = readResource("pixi.toml");
                logger.info("Building Appose (pixi) environment for Cellpose; first run may take a while");
                try {
                    environment = Appose.pixi()
                            .content(pixiToml)
                            .subscribeProgress((title, current, maximum) ->
                                    logger.info("Cellpose env build: {} ({}/{})", title, current, maximum))
                            .subscribeOutput(line -> logger.info("[pixi] {}", line))
                            .subscribeError(line -> logger.warn("[pixi] {}", line))
                            .build();
                } catch (BuildException e) {
                    throw new IOException("Failed to build the Appose (pixi) environment for Cellpose", e);
                }
            }
            return environment;
        }
    }

    /**
     * Name of the pixi sub-environment to activate for a given model family. The CUDA suffix mirrors
     * the imglib2-cellpose scheme: {@code cu126} when an NVIDIA GPU appears to be available,
     * {@code cpu} otherwise (and always {@code cpu} on macOS).
     *
     * @param cellposeSam true for the Cellpose-SAM (Cellpose 4) family, false for Cellpose 3
     * @return the sub-environment name, e.g. {@code cp3-cpu} or {@code cp4-cu126}
     */
    static String envName(boolean cellposeSam) {
        String family = cellposeSam ? "cp4" : "cp3";
        return family + "-" + (hasCuda() ? "cu126" : "cpu");
    }

    /**
     * Read one of the vendored Appose resources (a script or the pixi manifest) as a UTF-8 string.
     *
     * @param name the resource file name, relative to {@link #RESOURCE_ROOT}
     * @return the resource content
     * @throws IOException if the resource is missing or cannot be read
     */
    static String readResource(String name) throws IOException {
        String path = RESOURCE_ROOT + name;
        try (InputStream stream = ApposeEnvironments.class.getResourceAsStream(path)) {
            if (stream == null)
                throw new IOException("Missing bundled Appose resource: " + path);
            return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    /**
     * Best-effort check for an available NVIDIA GPU by trying to run {@code nvidia-smi}. Returns
     * false on macOS. Adapted, with attribution, from imglib2-cellpose (BSD-3-Clause).
     *
     * @return true if an NVIDIA GPU appears to be available
     */
    private static boolean hasCuda() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("mac") || os.contains("darwin"))
            return false;
        try {
            ProcessBuilder pb = new ProcessBuilder("nvidia-smi");
            pb.redirectErrorStream(true);
            Process process = pb.start();
            process.waitFor();
            return process.exitValue() == 0;
        } catch (IOException | InterruptedException e) {
            return false;
        }
    }
}
