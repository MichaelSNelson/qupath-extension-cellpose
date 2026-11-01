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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Watches the pixi processes that build or activate the Cellpose environment, so a slow build
 * does not look like a hung one and an interrupted build does not leave pixi running.
 *
 * <p>Written after a first build on a cluster with a CephFS home directory, where unpacking the
 * PyTorch environment (tens of thousands of small files) took many minutes with no output, and
 * stopping the script left pixi running, so the next attempt sat waiting on its lock.
 *
 * <ul>
 *   <li>At the start, any pixi already working on the same environment directory is reported. If
 *       it is a child of this QuPath it can only be left over from a stopped build (builds are
 *       serialized), so it is ended; one started by another process is reported, not touched,
 *       because on shared storage another job may be building legitimately.</li>
 *   <li>Every 30 s while it runs, a log line says how long it has taken and how much pixi has read
 *       and written (from {@code /proc/<pid>/io} on Linux), and warns when nothing moves while
 *       another pixi holds the same directory.</li>
 *   <li>When the step fails or is interrupted, the pixi it started is ended, as is any still
 *       running when QuPath exits.</li>
 * </ul>
 *
 * Our pixi installs are recognised by their command line: {@code pixi install} with
 * {@code --manifest-path <envDir>/pixi.toml}. The worker's own long-lived {@code pixi run} is never
 * matched.
 */
final class PixiWatch implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(PixiWatch.class);

    static final long HEARTBEAT_SECONDS = 30;

    /** Environment directories with a watched step in progress, for the shutdown hook. */
    private static final Set<String> ACTIVE = ConcurrentHashMap.newKeySet();
    private static volatile boolean shutdownHookInstalled;

    private static final Pattern IO_LINE = Pattern.compile("^(rchar|wchar):\\s*(\\d+)\\s*$", Pattern.MULTILINE);
    private static final Pattern LEVEL = Pattern.compile("^\\s*(TRACE|DEBUG|INFO|WARN|ERROR)\\b");

    private final String envDir;
    private final String what;
    private final long startNanos = System.nanoTime();
    private final ScheduledExecutorService timer;
    private long lastBytes = -1;
    private int stillBeats;
    private boolean succeeded;

    /**
     * Start watching one pixi step.
     *
     * @param envDir the environment directory pixi works in
     * @param what   what the step is, for the log ("Building the Cellpose environment")
     */
    PixiWatch(Path envDir, String what) {
        this(envDir, what, HEARTBEAT_SECONDS * 1000);
    }

    /** As above, with the progress interval in milliseconds (tests use a short one). */
    PixiWatch(Path envDir, String what, long periodMillis) {
        this.envDir = envDir.toAbsolutePath().normalize().toString();
        this.what = what;
        installShutdownHook();
        log(endOrReportLeftovers());
        ACTIVE.add(this.envDir);
        this.timer = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "Cellpose-pixi-watch");
            t.setDaemon(true);
            return t;
        });
        timer.scheduleAtFixedRate(() -> {
            try {
                log(beat());
            } catch (RuntimeException e) {
                logger.debug("pixi progress check failed: {}", e.getMessage());
            }
        }, periodMillis, periodMillis, TimeUnit.MILLISECONDS);
    }

    /** Mark the step as completed, so closing does not end its (already finished) pixi. */
    void succeeded() {
        succeeded = true;
    }

    @Override
    public void close() {
        timer.shutdownNow();
        ACTIVE.remove(envDir);
        if (!succeeded || Thread.currentThread().isInterrupted()) {
            List<ProcessHandle> ours = ourPixi();
            if (!ours.isEmpty())
                logger.info("{} did not complete; ending the pixi process(es) it started: {}", what,
                        ours.stream().map(h -> String.valueOf(h.pid())).toList());
            ours.forEach(PixiWatch::end);
        }
    }

    // ------------------------------------------------------------------ leftovers

    /** Ends this QuPath's leftover pixi installs on the directory; returns WARN messages. */
    List<String> endOrReportLeftovers() {
        List<String> messages = new java.util.ArrayList<>();
        for (ProcessHandle h : pixiOn(envDir)) {
            String started = h.info().startInstant().map(PixiWatch::time).orElse("unknown time");
            if (isOurs(h)) {
                messages.add("WARN:A pixi process (pid " + h.pid() + ", started " + started + ") from an earlier "
                        + "Cellpose run that was stopped is still working on " + envDir
                        + "; ending it so this run does not wait behind it");
                end(h);
            } else {
                messages.add("WARN:Another pixi process (pid " + h.pid() + ", started " + started + ") is already "
                        + "working on " + envDir + ". This run will wait until it finishes. If it is left over from "
                        + "a run that was stopped, end it (kill " + h.pid() + ").");
            }
        }
        return messages;
    }

    private static void log(List<String> messages) {
        for (String m : messages) {
            if (m.startsWith("WARN:"))
                logger.warn(m.substring(5));
            else
                logger.info(m.substring(m.indexOf(':') + 1));
        }
    }

    // ------------------------------------------------------------------ heartbeat

    /** One progress check; returns INFO/WARN messages. */
    List<String> beat() {
        List<String> messages = new java.util.ArrayList<>();
        long seconds = (System.nanoTime() - startNanos) / 1_000_000_000L;
        long bytes = -1;
        long read = 0, written = 0;
        for (ProcessHandle h : ourPixi()) {
            long[] io = readIo(h.pid());
            if (io != null) {
                read += io[0];
                written += io[1];
                bytes = Math.max(bytes, 0) + io[0] + io[1];
            }
        }
        if (bytes >= 0)
            messages.add("INFO:" + what + ": still working after " + seconds + " s; pixi has read "
                    + read / 1_000_000 + " MB and written " + written / 1_000_000 + " MB so far");
        else
            messages.add("INFO:" + what + ": still working after " + seconds + " s");
        // only where the I/O counters exist: an unreadable /proc must not read as 'no progress'
        if (bytes >= 0 && bytes == lastBytes) {
            stillBeats++;
            if (stillBeats == 2) {
                List<ProcessHandle> others = pixiOn(envDir).stream().filter(h -> !isOurs(h)).toList();
                if (!others.isEmpty())
                    messages.add("WARN:" + what + ": no progress for a minute, and another pixi process ("
                            + others.stream().map(h -> "pid " + h.pid()).toList() + ") is working on " + envDir
                            + ". This run is waiting for it. If it is left over from a stopped run, end it.");
                else
                    messages.add("WARN:" + what + ": no progress for a minute. On a slow or network filesystem "
                            + "this can be normal; if it lasts, check the node's network access and the free space in "
                            + envDir);
            }
        } else {
            stillBeats = 0;
        }
        lastBytes = bytes;
        return messages;
    }

    // ------------------------------------------------------------------ helpers

    /** Pixi processes whose command line names this environment directory. */
    private static List<ProcessHandle> pixiOn(String envDir) {
        return ProcessHandle.allProcesses()
                .filter(ProcessHandle::isAlive)
                .filter(h -> isPixiFor(h.info().command().orElse(""), h.info().commandLine().orElse(""), envDir))
                .toList();
    }

    /** This QuPath's pixi processes for this environment directory. */
    private List<ProcessHandle> ourPixi() {
        return ProcessHandle.current().descendants()
                .filter(ProcessHandle::isAlive)
                .filter(h -> isPixiFor(h.info().command().orElse(""), h.info().commandLine().orElse(""), envDir))
                .toList();
    }

    /**
     * Whether this is a pixi INSTALL working on {@code envDir}. Only installs: the Python worker
     * itself runs as {@code pixi run --manifest-path <envDir>/pixi.toml ...} for the whole session,
     * and must never be taken for a leftover.
     */
    static boolean isPixiFor(String command, String commandLine, String envDir) {
        String name = Path.of(command.isEmpty() ? "x" : command).getFileName().toString();
        return (name.equals("pixi") || name.equals("pixi.exe"))
                && commandLine.matches(".*\\sinstall(\\s.*|$)")
                && commandLine.replace('\\', '/').contains(Path.of(envDir, "pixi.toml").toString().replace('\\', '/'));
    }

    private static boolean isOurs(ProcessHandle h) {
        long me = ProcessHandle.current().pid();
        for (Optional<ProcessHandle> p = h.parent(); p.isPresent(); p = p.get().parent()) {
            if (p.get().pid() == me)
                return true;
        }
        return false;
    }

    private static void end(ProcessHandle h) {
        h.destroy();
        try {
            h.onExit().get(5, TimeUnit.SECONDS);
        } catch (Exception e) {
            h.destroyForcibly();
        }
    }

    /** {@code {rchar, wchar}} from {@code /proc/<pid>/io}, or null where it cannot be read. */
    private static long[] readIo(long pid) {
        Path io = Path.of("/proc", String.valueOf(pid), "io");
        if (!Files.isReadable(io))
            return null;
        try {
            return parseIo(Files.readString(io));
        } catch (IOException e) {
            return null;
        }
    }

    static long[] parseIo(String text) {
        long r = -1, w = -1;
        Matcher m = IO_LINE.matcher(text);
        while (m.find()) {
            if (m.group(1).equals("rchar"))
                r = Long.parseLong(m.group(2));
            else
                w = Long.parseLong(m.group(2));
        }
        return r < 0 || w < 0 ? null : new long[] {r, w};
    }

    /**
     * The log level pixi gave a line ({@code DEBUG}, {@code INFO}, ...), or null when it gave none.
     * pixi writes everything, debug output included, to stderr, so the stream says nothing about
     * how serious a line is.
     */
    static String levelOf(String line) {
        Matcher m = LEVEL.matcher(line);
        return m.find() ? m.group(1) : null;
    }

    private static String time(Instant instant) {
        return DateTimeFormatter.ofPattern("HH:mm:ss").withZone(ZoneId.systemDefault()).format(instant);
    }

    private static synchronized void installShutdownHook() {
        if (shutdownHookInstalled)
            return;
        shutdownHookInstalled = true;
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            // A child process outlives QuPath on Linux and macOS; a half-finished install left
            // running then blocks the next QuPath's build on the same directory.
            for (String dir : ACTIVE)
                ProcessHandle.current().descendants()
                        .filter(h -> isPixiFor(h.info().command().orElse(""), h.info().commandLine().orElse(""), dir))
                        .forEach(ProcessHandle::destroyForcibly);
        }, "Cellpose-pixi-shutdown"));
    }
}
