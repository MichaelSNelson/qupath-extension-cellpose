package qupath.ext.biop.cellpose.backend;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PixiWatchTest {

    @TempDir
    Path dir;

    @Test
    void pixiLevelsAreReadFromTheLineNotTheStream() {
        assertEquals("DEBUG", PixiWatch.levelOf("DEBUG pixi_config: Loading config from /etc/pixi/config.toml"));
        assertEquals("INFO", PixiWatch.levelOf(" INFO pixi_core::lock_file::update: the lock-file is up-to-date"));
        assertEquals("WARN", PixiWatch.levelOf(" WARN something"));
        assertNull(PixiWatch.levelOf("Downloading torch-2.10.0 (900 MB)"));
    }

    @Test
    void procIoIsParsed() {
        String io = "rchar: 1048576\nwchar: 2097152\nsyscr: 10\nsyscw: 20\nread_bytes: 0\nwrite_bytes: 4096\n";
        assertArrayEquals(new long[] {1048576, 2097152}, PixiWatch.parseIo(io));
        assertNull(PixiWatch.parseIo("garbage"));
    }

    @Test
    void ourPixiIsRecognisedByTheManifestPathOnItsCommandLine() {
        String env = "/home/u/.local/share/appose/qupath-cellpose-appose";
        assertTrue(PixiWatch.isPixiFor("/home/u/.local/share/appose/.pixi/bin/pixi",
                "/home/u/.local/share/appose/.pixi/bin/pixi install --manifest-path " + env + "/pixi.toml", env));
        // another extension's environment
        assertFalse(PixiWatch.isPixiFor("/x/pixi", "/x/pixi install --manifest-path /home/u/.local/share/appose/qupath-qpcat/pixi.toml", env));
        // not pixi
        assertFalse(PixiWatch.isPixiFor("/usr/bin/python", "python --manifest-path " + env + "/pixi.toml", env));
        // the running Python worker: pixi run, never a leftover
        assertFalse(PixiWatch.isPixiFor("/x/pixi", "/x/pixi run --manifest-path " + env
                + "/pixi.toml --environment cp3-cu126 python -c import appose", env));
    }

    /** A copy of bash named pixi, run so its command line names {@code env}, like the real one. */
    private Process fakePixi(Path env) throws Exception {
        return fakePixi(env, "install");
    }

    private Process fakePixi(Path env, String subcommand) throws Exception {
        Path pixi = dir.resolve("bin").resolve("pixi");
        Files.createDirectories(pixi.getParent());
        if (!Files.exists(pixi)) {
            Files.copy(Path.of("/bin/bash"), pixi, StandardCopyOption.COPY_ATTRIBUTES);
            pixi.toFile().setExecutable(true);
        }
        Process p = new ProcessBuilder(pixi.toString(), "-c", "sleep 60; true", subcommand, "--manifest-path",
                env.resolve("pixi.toml").toString()).start();
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (!p.toHandle().info().commandLine().orElse("").contains("--manifest-path") && System.nanoTime() < deadline)
            Thread.sleep(50);
        return p;
    }

    @Test
    @EnabledOnOs(OS.LINUX)
    void aLeftoverPixiFromThisQuPathIsEndedWhenTheNextStepStarts() throws Exception {
        Path env = Files.createDirectories(dir.resolve("env"));
        Process leftover = fakePixi(env);
        assertTrue(leftover.isAlive());
        try (PixiWatch watch = new PixiWatch(env, "test")) {
            watch.succeeded();
            assertTrue(leftover.waitFor(10, TimeUnit.SECONDS), "the leftover pixi should have been ended");
        }
    }

    @Test
    @EnabledOnOs(OS.LINUX)
    void aStepThatDoesNotCompleteEndsThePixiItStarted() throws Exception {
        Path env = Files.createDirectories(dir.resolve("env2"));
        Process started;
        try (PixiWatch watch = new PixiWatch(env, "test")) {
            started = fakePixi(env); // started by the step, after the watch began
            // no succeeded(): the step failed or was interrupted
        }
        assertTrue(started.waitFor(10, TimeUnit.SECONDS), "close() should have ended the step's pixi");
    }

    @Test
    @EnabledOnOs(OS.LINUX)
    void progressReportsWhatPixiHasWritten() throws Exception {
        Path env = Files.createDirectories(dir.resolve("env5"));
        Path pixi = dir.resolve("bin").resolve("pixi");
        fakePixi(Files.createDirectories(dir.resolve("warmup"))).destroyForcibly(); // creates the fake binary
        try (PixiWatch watch = new PixiWatch(env, "Installing test", 3_600_000)) {
            Process p = new ProcessBuilder(pixi.toString(), "-c",
                    // printf is a bash builtin, so the fake pixi itself does the writing, as pixi does
                    "exec 3> '" + dir.resolve("out.bin") + "'; for i in $(seq 40); do printf '%1000000s' '' >&3; done; sleep 60; true",
                    "install", "--manifest-path", env.resolve("pixi.toml").toString()).start();
            try {
                Thread.sleep(1500);
                String line = String.join("\n", watch.beat());
                assertTrue(line.startsWith("INFO:Installing test: still working after"), line);
                java.util.regex.Matcher m = java.util.regex.Pattern.compile("written (\\d+) MB").matcher(line);
                assertTrue(m.find(), line);
                assertTrue(Integer.parseInt(m.group(1)) >= 39, line);
            } finally {
                p.destroyForcibly();
            }
            watch.succeeded();
        }
    }

    @Test
    @EnabledOnOs(OS.LINUX)
    void aPixiStartedByAnotherProcessIsReportedNotEnded() throws Exception {
        Path env = Files.createDirectories(dir.resolve("env6"));
        Path pixi = dir.resolve("bin").resolve("pixi");
        fakePixi(Files.createDirectories(dir.resolve("warmup2"))).destroyForcibly();
        // double fork: the fake pixi is re-parented away from this JVM, like another job's pixi
        new ProcessBuilder("bash", "-c", "setsid '" + pixi + "' -c 'sleep 30; true' install --manifest-path '"
                + env.resolve("pixi.toml") + "' > /dev/null 2>&1 &").start().waitFor();
        Thread.sleep(500);
        try (PixiWatch watch = new PixiWatch(env, "test", 3_600_000)) {
            String report = String.join("\n", watch.endOrReportLeftovers());
            assertTrue(report.contains("Another pixi process"), report);
            assertTrue(report.contains("kill "), report);
            watch.succeeded();
        } finally {
            new ProcessBuilder("pkill", "-f", env.resolve("pixi.toml").toString()).start().waitFor();
        }
    }

    @Test
    @EnabledOnOs(OS.LINUX)
    void theRunningWorkersPixiRunIsNeverEnded() throws Exception {
        Path env = Files.createDirectories(dir.resolve("env4"));
        Process worker = fakePixi(env, "run");
        try {
            try (PixiWatch watch = new PixiWatch(env, "test")) {
                // not completed: close() ends this step's installs, but must not touch the worker
            }
            Thread.sleep(500);
            assertTrue(worker.isAlive(), "the worker's pixi run must survive another step on the same environment");
        } finally {
            worker.destroyForcibly();
        }
    }

    @Test
    @EnabledOnOs(OS.LINUX)
    void aCompletedStepLeavesOtherPixiAlone() throws Exception {
        Path env = Files.createDirectories(dir.resolve("env3"));
        Path otherEnv = Files.createDirectories(dir.resolve("other"));
        Process other = fakePixi(otherEnv);
        try {
            try (PixiWatch watch = new PixiWatch(env, "test")) {
                watch.succeeded();
            }
            assertTrue(other.isAlive(), "a pixi working on another environment must not be touched");
        } finally {
            other.destroyForcibly();
        }
    }
}
