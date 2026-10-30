package qupath.ext.biop.cellpose.backend;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** A changed manifest replaces the environment without making the run wait for the old one. */
class EnvironmentReplacementTest {

    @TempDir
    Path env;

    private void installedEnvironment(String manifest) throws IOException {
        Path lib = Files.createDirectories(env.resolve(".pixi/envs/cp4-cu126/lib"));
        Files.writeString(lib.resolve("libtorch.so"), "x");
        Files.writeString(lib.resolve("libcrypto.so.3"), "x");
        Files.createSymbolicLink(lib.resolve("libcrypto.so"), Path.of("libcrypto.so.3"));
        Files.writeString(env.resolve("pixi.toml"), manifest);
    }

    private List<String> retired() throws IOException {
        try (var children = Files.list(env)) {
            return children.map(p -> p.getFileName().toString()).filter(n -> n.startsWith(".pixi-old-")).toList();
        }
    }

    @Test
    void aChangedManifestRetiresTheEnvironmentAndItIsDeletedInTheBackground() throws Exception {
        installedEnvironment("old");
        ApposeEnvironments.syncManifest(env, "new", "lock");
        assertFalse(Files.exists(env.resolve(".pixi")), "the build needs a clean .pixi straight away");
        assertEquals("new", Files.readString(env.resolve("pixi.toml")));
        long deadline = System.nanoTime() + 10_000_000_000L;
        while (!retired().isEmpty() && System.nanoTime() < deadline)
            Thread.sleep(50);
        assertTrue(retired().isEmpty(), "the retired environment should have been deleted");
    }

    @Test
    void anUnchangedManifestLeavesTheEnvironmentAlone() throws Exception {
        installedEnvironment("same");
        ApposeEnvironments.syncManifest(env, "same", "lock");
        assertTrue(Files.exists(env.resolve(".pixi/envs/cp4-cu126/lib/libtorch.so")));
        assertTrue(retired().isEmpty());
    }

    @Test
    void aDeletionInterruptedByQuPathExitingIsFinishedByTheNextRun() throws Exception {
        Files.createDirectories(env.resolve(".pixi-old-123/envs/x"));
        Files.writeString(env.resolve(".pixi-old-123/envs/x/file"), "x");
        Thread t = ApposeEnvironments.deleteRetiredEnvironments(env);
        assertNotNull(t);
        t.join(10_000);
        assertTrue(retired().isEmpty());
        assertNull(ApposeEnvironments.deleteRetiredEnvironments(env), "nothing left to delete");
    }
}
