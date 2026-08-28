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

import org.apposed.appose.Appose;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.biop.cellpose.CellposeExtension;
import qupath.ext.biop.cellpose.ui.PythonConsoleWindow;

import java.awt.GraphicsEnvironment;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.concurrent.TimeUnit;

/**
 * Builds and caches the Appose (pixi) environment used by the in-process {@link ApposeBackend}.
 * <p>
 * This is Apache-2.0 original work. The environment-build call and the Cellpose 3 vs Cellpose-SAM
 * (Cellpose 4) environment-name/CUDA-suffix scheme are adapted, with attribution, from the
 * BSD-3-Clause imglib2-cellpose project ({@code net.imglib2.cellpose.Cellpose} and
 * {@code CellposeRunner}; see NOTICE). The pixi manifest is a single vendored {@code pixi.toml}
 * declaring the {@code cp3-*}/{@code cp4-*} sub-environments; the two Cellpose major versions
 * cannot share one environment, so callers activate the sub-environment matching the model family.
 * <p>
 * The lockfile staging, GPU-detection hardening, first-run warning, and Windows file-lock recovery
 * follow the lab's documented Appose conventions.
 */
final class ApposeEnvironments {

    private static final Logger logger = LoggerFactory.getLogger(ApposeEnvironments.class);

    /** Root of the vendored Appose resources (scripts and pixi manifest). */
    static final String RESOURCE_ROOT = "/qupath/ext/biop/cellpose/appose/";

    /**
     * Directory name under the Appose base into which the environment is installed. This must match
     * the {@code name} field of the vendored {@code pixi.toml}, since Appose derives the env dir from
     * it. Used only to stage the manifest/lock (see {@link #syncManifest}) and to detect the first
     * build; the build itself still targets the same directory.
     */
    /**
     * Directory (under Appose's share dir) holding this extension's Python environment.
     * <p>
     * Deliberately NOT the vendored manifest's own workspace name, {@code cellpose-appose}: that is
     * also what the Fiji cellpose-appose plugin uses, and Appose derives the environment directory
     * from the workspace name. Sharing it means both tools stage into one pixi workspace under
     * incompatible policies - this extension installs a pinned, committed lock while the Fiji plugin
     * resolves fresh - so each would overwrite the other's manifest and trigger a multi-GB rebuild.
     * The name is overridden on the builder rather than in the manifest, which stays byte-identical
     * to upstream.
     */
    private static final String ENV_DIR_NAME = "qupath-cellpose-appose";

    private static final Object LOCK = new Object();
    private static volatile Environment environment;
    private static volatile boolean firstRunWarned = false;

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
                String pixiLock = readResource("pixi.lock");

                // Ask before the first build, not after: the environment is several GB, and once it
                // is downloaded to the wrong drive the only remedy is to delete it and start over.
                // Only when nothing is configured and nothing is built yet -- never re-ask, and
                // never override a choice the user already made in the preferences.
                promptForEnvironmentLocationOnFirstRun();

                Path envDir = getEnvironmentPath();
                boolean firstBuild = !Files.exists(envDir.resolve(".pixi"));
                warnIfUnsuitable(getEnvironmentBase());

                // Stage the manifest AND the committed lock, so every user installs the exact
                // dependency tree this was tested against instead of re-resolving. The lock is
                // generated by the same pixi version Appose bundles (v0.58.0 -> lock format v6):
                // a lock written by a newer pixi is format v7, which the bundled pixi rejects with
                // "lock file version 7 ... only up to 6 supported", failing every install. See
                // .github/workflows or the README for how to regenerate it.
                syncManifest(envDir, pixiToml, pixiLock);

                if (firstBuild)
                    warnFirstRun();

                environment = buildEnvironment(pixiToml, envDir, firstBuild);
            }
            return environment;
        }
    }

    /**
     * Build the environment, routing pixi progress/output to the log and the Python console. On a
     * lock-format mismatch (a user whose Appose bundles an older pixi that cannot read our committed
     * lock format) the staged lock is deleted and the build retried once, resolving fresh. A Windows
     * file lock is surfaced with manual recovery steps and never auto-wiped.
     */
    private static Environment buildEnvironment(String pixiToml, Path envDir, boolean firstBuild) throws IOException {
        if (firstBuild) {
            String bar = "*".repeat(78);
            logger.info(bar);
            logger.info("**  BUILDING THE CELLPOSE PYTHON ENVIRONMENT (first run only)");
            logger.info("**  Downloading a multi-GB PyTorch/Cellpose environment. This can take");
            logger.info("**  SEVERAL MINUTES. QuPath is NOT frozen -- watch progress in");
            logger.info("**  Extensions > Cellpose > Python console.");
            logger.info(bar);
        } else {
            logger.info("Preparing the Cellpose Appose environment...");
        }
        try {
            return runBuild(pixiToml);
        } catch (BuildException e) {
            String causes = collectCauseMessages(e);
            if (looksLikeWindowsFileLock(causes)) {
                surfaceWindowsFileLock(envDir);
                throw new IOException("Cellpose environment build blocked by a Windows file lock; "
                        + "see the recovery steps in the log/notification", e);
            }
            if (looksLikeLockFormatSkew(causes)) {
                logger.warn("The staged pixi.lock could not be read by this pixi (format skew); "
                        + "deleting it and re-resolving once. Cause: {}", causes);
                try {
                    Files.deleteIfExists(envDir.resolve("pixi.lock"));
                } catch (IOException ignore) {
                    // best effort; the retry below will surface any real problem
                }
                try {
                    return runBuild(pixiToml);
                } catch (BuildException e2) {
                    String causes2 = collectCauseMessages(e2);
                    if (looksLikeWindowsFileLock(causes2)) {
                        surfaceWindowsFileLock(envDir);
                        throw new IOException("Cellpose environment build blocked by a Windows file lock; "
                                + "see the recovery steps in the log/notification", e2);
                    }
                    throw new IOException("Failed to build the Appose (pixi) environment for Cellpose "
                            + "(after re-resolving without the staged lock)", e2);
                }
            }
            throw new IOException("Failed to build the Appose (pixi) environment for Cellpose", e);
        }
    }

    private static Environment runBuild(String pixiToml) throws BuildException {
        try {
            return withExtensionClassLoader(() -> Appose.pixi()
                    .content(pixiToml)
                    // base() sets Appose's envDir FIELD -- the environment directory itself, not
                    // a parent to append name() to (verified in BaseBuilder bytecode: base ->
                    // envDir, name -> envName, and envName is only consulted when envDir is unset).
                    // So pass the full path, which is also what stages the manifest and lock, and
                    // do not set name() as well: passing a parent here builds the environment one
                    // level up from where the manifest was staged, which then silently resolves a
                    // second copy.
                    .base(getEnvironmentPath().toFile())
                    .subscribeProgress((title, current, maximum) -> {
                        String line = "Cellpose env build: " + title + " (" + current + "/" + maximum + ")";
                        logger.info(line);
                        PythonConsoleWindow.appendMessage(line);
                    })
                    .subscribeOutput(line -> {
                        logger.info("[pixi] {}", line);
                        PythonConsoleWindow.appendMessage("[pixi] " + line);
                    })
                    .subscribeError(line -> {
                        logger.warn("[pixi] {}", line);
                        PythonConsoleWindow.appendMessage("[pixi] " + line);
                    })
                    .build());
        } catch (BuildException e) {
            throw e;
        } catch (Exception e) {
            throw new BuildException("Unexpected error building the Cellpose Appose environment", e);
        }
    }

    /**
     * Stage the manifest and the committed lock into the environment directory before building, so
     * the install reproduces the tested dependency tree rather than re-resolving. If either changed
     * relative to what is on disk, the existing {@code .pixi} env is wiped to force a clean
     * reinstall. Any staged lock the bundled pixi cannot read is cleared first, so a lock left by an
     * older build of this extension cannot break the install.
     *
     * @param envDir       the environment directory
     * @param expectedToml the bundled manifest content
     * @param expectedLock the bundled lock content
     */
    private static void syncManifest(Path envDir, String expectedToml, String expectedLock) throws IOException {
        Path tomlFile = envDir.resolve("pixi.toml");
        Path lockFile = envDir.resolve("pixi.lock");

        clearIncompatibleLock(lockFile);
        Files.createDirectories(envDir);

        boolean changed = !Files.exists(tomlFile)
                || !normalize(Files.readString(tomlFile, StandardCharsets.UTF_8)).equals(normalize(expectedToml));
        if (changed && Files.exists(tomlFile))
            deletePixiDir(envDir.resolve(".pixi"));

        Files.writeString(tomlFile, expectedToml, StandardCharsets.UTF_8);
        Files.writeString(lockFile, expectedLock, StandardCharsets.UTF_8);
    }

    /**
     * Delete an on-disk {@code pixi.lock} whose format is too new for the bundled pixi (version 7+),
     * such as one a previous build of this extension staged. A lock the local pixi wrote itself
     * (version 6 or lower) is left in place so it can be reused. Best-effort; a malformed lock is
     * removed so the build can proceed.
     */
    private static void clearIncompatibleLock(Path lockFile) {
        try {
            if (!Files.exists(lockFile))
                return;
            for (String line : Files.readAllLines(lockFile, StandardCharsets.UTF_8)) {
                String t = line.trim();
                if (t.startsWith("version:")) {
                    int v = Integer.parseInt(t.substring("version:".length()).trim());
                    if (v >= 7)
                        Files.deleteIfExists(lockFile);
                    return;
                }
            }
        } catch (Exception e) {
            try {
                Files.deleteIfExists(lockFile);
            } catch (IOException ignore) {
                // best effort
            }
        }
    }

    private static String normalize(String s) {
        return s.replace("\r\n", "\n").strip();
    }

    /**
     * Best-effort recursive delete of the {@code .pixi} directory to force a clean reinstall. On
     * Windows a locked file can block deletion; a rename-then-delete fallback usually still succeeds.
     */
    private static void deletePixiDir(Path pixiDir) throws IOException {
        if (!Files.exists(pixiDir))
            return;
        try {
            deleteRecursively(pixiDir);
        } catch (IOException e) {
            Path renamed = pixiDir.resolveSibling(".pixi-old-" + System.currentTimeMillis());
            try {
                Files.move(pixiDir, renamed);
                deleteRecursively(renamed);
            } catch (IOException e2) {
                throw new IOException("Could not remove the existing Cellpose environment at " + pixiDir, e2);
            }
        }
    }

    private static void deleteRecursively(Path path) throws IOException {
        if (!Files.exists(path))
            return;
        try (var walk = Files.walk(path)) {
            walk.sorted((a, b) -> b.getNameCount() - a.getNameCount())
                    .forEach(p -> {
                        try {
                            Files.deleteIfExists(p);
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    });
        } catch (RuntimeException e) {
            if (e.getCause() instanceof IOException io)
                throw io;
            throw e;
        }
    }

    /**
     * On the very first build, offer to put the multi-GB environment somewhere other than the
     * default. Does nothing when a location is already configured, when an environment already
     * exists anywhere we would look, or when there is no GUI to ask through -- a scripted or
     * headless run must never block on a dialog.
     */
    private static void promptForEnvironmentLocationOnFirstRun() {
        if (!CellposeExtension.getApposeEnvDirPreference().isEmpty())
            return; // the user has already chosen
        if (Files.exists(getEnvironmentPath().resolve(".pixi")))
            return; // already built in the default location; leave it alone
        if (GraphicsEnvironment.isHeadless())
            return;

        try {
            Path defaultBase = getEnvironmentBase();
            boolean choose = qupath.fx.dialogs.Dialogs.showYesNoDialog(
                    "Cellpose",
                    "The in-process (Appose) backend needs to build a Python environment of several "
                            + "gigabytes.\n\nIt will go in:\n" + defaultBase + "\n\n"
                            + "Choose a different location?\n\n"
                            + "This is worth doing on a shared workstation, where every user would "
                            + "otherwise get their own copy on the system drive. You can change it later "
                            + "in Edit > Preferences > Cellpose.");
            if (!choose)
                return;

            File chosen = qupath.fx.dialogs.FileChoosers.promptForDirectory(
                    "Where should the Cellpose Python environment go?", defaultBase.toFile());
            if (chosen == null)
                return; // cancelled: fall through to the default

            CellposeExtension.setApposeEnvDirPreference(chosen.getAbsolutePath());
            logger.info("Cellpose Appose environment directory set to {}", chosen.getAbsolutePath());
        } catch (RuntimeException e) {
            // Never let the prompt itself stop a run; the default location is always usable.
            logger.debug("Could not prompt for the environment location: {}", e.getMessage());
        }
    }

    /**
     * @return the directory that will CONTAIN the environment: the user's chosen location, or the
     * Appose default when the preference is empty
     */
    static Path getEnvironmentBase() {
        String configured = CellposeExtension.getApposeEnvDirPreference();
        if (!configured.isEmpty())
            return Paths.get(configured);
        return Paths.get(System.getProperty("user.home"), ".local", "share", "appose");
    }

    /** @return the directory Appose installs the Cellpose environment into. */
    private static Path getEnvironmentPath() {
        return getEnvironmentBase().resolve(ENV_DIR_NAME);
    }

    /**
     * Warn about a base directory pixi cannot build in. A path containing spaces breaks
     * {@code pixi run --manifest-path}, and the resulting failure names neither the path nor the
     * space, so it is worth saying plainly before the multi-GB download rather than after.
     *
     * @param base the chosen containing directory
     */
    private static void warnIfUnsuitable(Path base) {
        if (base.toString().contains(" ")) {
            logger.warn("The Cellpose Appose environment directory contains a space: {}. "
                    + "pixi cannot build in such a path; choose a directory without spaces in "
                    + "Edit > Preferences > Cellpose.", base);
        }
    }

    /**
     * Resolve whether the in-process backend should run on the GPU for the requested device. For
     * {@link CellposeDevice#AUTO}, GPU detection ({@link #hasCuda()}) decides; {@link CellposeDevice#GPU}
     * and {@link CellposeDevice#CPU} force the choice.
     *
     * @param device the requested device (never null)
     * @return true to activate the CUDA sub-environment and request GPU execution
     */
    static boolean resolveUseGpu(CellposeDevice device) {
        switch (device) {
            case GPU:
                return true;
            case CPU:
                return false;
            case AUTO:
            default:
                return hasCuda();
        }
    }

    /**
     * Name of the pixi sub-environment to activate for a given model family and resolved device. The
     * CUDA suffix mirrors the imglib2-cellpose scheme: {@code cu126} when GPU execution is requested,
     * {@code cpu} otherwise.
     *
     * @param cellposeSam true for the Cellpose-SAM (Cellpose 4) family, false for Cellpose 3
     * @param useGpu      the resolved GPU/CPU choice (see {@link #resolveUseGpu(CellposeDevice)})
     * @return the sub-environment name, e.g. {@code cp3-cpu} or {@code cp4-cu126}
     */
    static String envName(boolean cellposeSam, boolean useGpu) {
        String family = cellposeSam ? "cp4" : "cp3";
        return family + "-" + (useGpu ? "cu126" : "cpu");
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
     * Swap the thread context classloader to the extension's classloader while running the given
     * body, restoring it afterwards. Appose's {@code ServiceLoader}-based plugin lookups (shared
     * memory, Groovy JSON) use the TCCL, which on QuPath worker threads cannot otherwise see the
     * registrations bundled in the shaded extension JAR.
     *
     * @param body the work to run
     * @param <T>  the result type
     * @return the body's result
     * @throws Exception whatever the body throws
     */
    static <T> T withExtensionClassLoader(Callable<T> body) throws Exception {
        Thread thread = Thread.currentThread();
        ClassLoader previous = thread.getContextClassLoader();
        thread.setContextClassLoader(ApposeEnvironments.class.getClassLoader());
        try {
            return body.call();
        } finally {
            thread.setContextClassLoader(previous);
        }
    }

    /**
     * Robust best-effort check for an available NVIDIA GPU. Tries {@code nvidia-smi} on the PATH,
     * and on Windows additionally the canonical {@code System32} path and whatever {@code where
     * nvidia-smi} resolves. A CUDA GPU is considered present if any invocation exits 0 or prints a
     * recognizable driver line. Always false on macOS (no NVIDIA/CUDA). Adapted, with attribution,
     * from imglib2-cellpose (BSD-3-Clause) and hardened per the lab's Windows notes.
     *
     * @return true if an NVIDIA GPU appears to be available
     */
    private static boolean hasCuda() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (os.contains("mac") || os.contains("darwin"))
            return false;
        boolean windows = os.contains("win");

        if (detectCuda(List.of("nvidia-smi")))
            return true;
        if (windows) {
            if (detectCuda(List.of("C:\\Windows\\System32\\nvidia-smi.exe")))
                return true;
            String resolved = whereWindows("nvidia-smi");
            if (resolved != null && detectCuda(List.of(resolved)))
                return true;
        }
        return false;
    }

    /** Run a candidate nvidia-smi command; true if it exits 0 or prints a recognizable driver line. */
    private static boolean detectCuda(List<String> command) {
        try {
            ProcessBuilder pb = new ProcessBuilder(command);
            pb.redirectErrorStream(true);
            Process process = pb.start();
            String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            if (!process.waitFor(15, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                return false;
            }
            boolean driverLine = output.contains("Driver Version")
                    || output.contains("CUDA Version")
                    || output.contains("NVIDIA-SMI");
            return process.exitValue() == 0 || driverLine;
        } catch (IOException | InterruptedException e) {
            if (e instanceof InterruptedException)
                Thread.currentThread().interrupt();
            return false;
        }
    }

    /** Resolve a command's path via {@code where} on Windows; null if not found. */
    private static String whereWindows(String command) {
        try {
            ProcessBuilder pb = new ProcessBuilder("where", command);
            pb.redirectErrorStream(true);
            Process process = pb.start();
            String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            if (!process.waitFor(15, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                return null;
            }
            if (process.exitValue() != 0)
                return null;
            for (String line : output.split("\\R")) {
                String trimmed = line.strip();
                if (!trimmed.isEmpty())
                    return trimmed;
            }
            return null;
        } catch (IOException | InterruptedException e) {
            if (e instanceof InterruptedException)
                Thread.currentThread().interrupt();
            return null;
        }
    }

    /** Concatenate the full cause chain's messages so signature checks see nested causes. */
    static String collectCauseMessages(Throwable t) {
        StringBuilder sb = new StringBuilder();
        Throwable current = t;
        int guard = 0;
        while (current != null && guard++ < 50) {
            if (current.getMessage() != null)
                sb.append(current.getMessage()).append('\n');
            if (current.getCause() == current)
                break;
            current = current.getCause();
        }
        return sb.toString();
    }

    /** True if the message chain looks like a Windows conda-link file lock during (re)install. */
    static boolean looksLikeWindowsFileLock(String messages) {
        if (messages == null)
            return false;
        String m = messages.toLowerCase(Locale.ROOT);
        return m.contains("being used by another process")
                || m.contains("os error 32")
                || m.contains("failed to link");
    }

    /** True if the message chain looks like pixi failing to parse the staged lock's format. */
    static boolean looksLikeLockFormatSkew(String messages) {
        if (messages == null)
            return false;
        String m = messages.toLowerCase(Locale.ROOT);
        return m.contains("lock file version")
                || m.contains("lockfile version")
                || m.contains("failed to parse") && m.contains("lock")
                || m.contains("unexpected") && m.contains("lock")
                || m.contains("could not parse") && m.contains("lock");
    }

    /** Warn the user, once, that the first run downloads a large Python environment. */
    private static void warnFirstRun() {
        if (firstRunWarned)
            return;
        firstRunWarned = true;
        String message = "Building the Cellpose Python environment on first run. This downloads a "
                + "multi-GB Python/PyTorch/Cellpose environment and may take several minutes. "
                + "Progress is shown in Extensions > Cellpose > Python console.";
        logger.info(message);
        if (!GraphicsEnvironment.isHeadless()) {
            try {
                qupath.fx.dialogs.Dialogs.showInfoNotification("Cellpose", message);
            } catch (RuntimeException e) {
                logger.debug("Could not show first-run notification: {}", e.getMessage());
            }
        }
    }

    /** Surface manual recovery steps for a Windows file lock; does not auto-wipe the environment. */
    private static void surfaceWindowsFileLock(Path envDir) {
        Path pixiDir = envDir.resolve(".pixi");
        String message = "The Cellpose environment build was blocked because a file inside\n"
                + pixiDir + "\nis in use by another process. To recover:\n"
                + "1. Fully quit QuPath.\n"
                + "2. In Task Manager, end any stray python.exe or pixi.exe processes.\n"
                + "3. Delete the folder " + pixiDir + "\n"
                + "4. Relaunch QuPath and run Cellpose again.";
        logger.warn(message);
        if (!GraphicsEnvironment.isHeadless()) {
            try {
                qupath.fx.dialogs.Dialogs.showWarningNotification("Cellpose environment locked", message);
            } catch (RuntimeException e) {
                logger.debug("Could not show file-lock notification: {}", e.getMessage());
            }
        }
    }
}
