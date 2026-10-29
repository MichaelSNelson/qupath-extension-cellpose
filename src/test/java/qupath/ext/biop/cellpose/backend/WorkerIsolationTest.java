package qupath.ext.biop.cellpose.backend;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** What the Cellpose worker inherits from the environment QuPath was started in. */
class WorkerIsolationTest {

    private static final Path PREFIX = Path.of("/home/u/.local/share/appose/qupath-cellpose-appose/.pixi/envs/cp4-cu126");

    /** QuPath launched inside Apptainer by a script that sets up DJL from another PyTorch env. */
    @Test
    void anotherEnvironmentsLibrariesAreDroppedAndTheDriverIsKept() {
        String inherited = String.join(":",
                "/home/u/miniconda3/envs/qupath-pytorch/lib/python3.13/site-packages/nvidia/cudnn/lib",
                "/home/u/miniconda3/envs/qupath-pytorch/lib/python3.13/site-packages/torch/lib",
                "/.singularity.d/libs");
        Map<String, String> vars = ApposeEnvironments.workerVariables(
                Map.of("LD_LIBRARY_PATH", inherited, "PYTHONPATH", "/home/u/other"), PREFIX, true,
                dir -> dir.equals("/.singularity.d/libs"));
        assertEquals(PREFIX.resolve("lib") + ":/.singularity.d/libs", vars.get("LD_LIBRARY_PATH"));
        assertEquals("", vars.get("PYTHONPATH"));
        assertEquals("1", vars.get("PYTHONNOUSERSITE"));
        assertFalse(vars.containsKey("PYTHONHOME"), "only emptied when QuPath had it");
    }

    @Test
    void withNothingInheritedTheEnvironmentsLibComesFirst() {
        Map<String, String> vars = ApposeEnvironments.workerVariables(Map.of(), PREFIX, true, dir -> false);
        assertEquals(PREFIX.resolve("lib").toString(), vars.get("LD_LIBRARY_PATH"));
    }

    @Test
    void libraryPathIsLeftAloneOffLinux() {
        Map<String, String> vars = ApposeEnvironments.workerVariables(
                Map.of("LD_LIBRARY_PATH", "/x"), PREFIX, false, dir -> false);
        assertNull(vars.get("LD_LIBRARY_PATH"));
        assertEquals("1", vars.get("PYTHONNOUSERSITE"));
    }

    @Test
    void aDriverTooOldForTheBuildIsDetected() {
        assertFalse(ApposeEnvironments.driverRuns("cu130", "570.133.20"));
        assertTrue(ApposeEnvironments.driverRuns("cu130", "580.65.06"));
        assertTrue(ApposeEnvironments.driverRuns("cu126", "570.133.20"));
        assertFalse(ApposeEnvironments.driverRuns("cu126", "470.256.02"));
        assertTrue(ApposeEnvironments.driverRuns("cu130", null), "unknown driver: do not block");
        assertTrue(ApposeEnvironments.driverRuns("cu130", "garbage"));
    }
}
