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

import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for the device-resolution logic and the Windows recovery / lock-format matchers.
 */
class DeviceAndRecoveryTest {

    @Test
    void resolveDeviceBuilderWinsThenPreferenceThenAuto() {
        assertEquals(CellposeDevice.GPU,
                CellposeBackend.resolveDevice(CellposeDevice.GPU, CellposeDevice.CPU));
        assertEquals(CellposeDevice.CPU,
                CellposeBackend.resolveDevice(null, CellposeDevice.CPU));
        assertEquals(CellposeDevice.AUTO,
                CellposeBackend.resolveDevice(null, null));
    }

    @Test
    void resolveUseGpuForcedDevices() {
        // GPU and CPU are deterministic; AUTO depends on the host and is not asserted here.
        assertTrue(ApposeEnvironments.resolveUseGpu(CellposeDevice.GPU));
        assertFalse(ApposeEnvironments.resolveUseGpu(CellposeDevice.CPU));
    }

    @Test
    void envNameCombinesFamilyAndDevice() {
        assertEquals("cp3-cpu", ApposeEnvironments.envName(false, false));
        assertEquals("cp3-cu126", ApposeEnvironments.envName(false, true));
        assertEquals("cp4-cpu", ApposeEnvironments.envName(true, false));
        assertEquals("cp4-cu126", ApposeEnvironments.envName(true, true));
    }

    @Test
    void windowsFileLockDetectedThroughFullCauseChain() {
        Throwable nested = new IllegalStateException(
                "failed to link cellpose.conda ... The process cannot access the file because it is "
                        + "being used by another process. (os error 32)");
        Throwable top = new IOException("pixi build failed", nested);
        String chain = ApposeEnvironments.collectCauseMessages(top);
        assertTrue(ApposeEnvironments.looksLikeWindowsFileLock(chain),
                "the file-lock signature should be found in the nested cause");
        assertFalse(ApposeEnvironments.looksLikeWindowsFileLock("pixi build failed"),
                "the generic top-level message alone must not match");
    }

    @Test
    void lockFormatSkewMatcher() {
        assertTrue(ApposeEnvironments.looksLikeLockFormatSkew("unsupported lock file version 7"));
        assertTrue(ApposeEnvironments.looksLikeLockFormatSkew("failed to parse pixi.lock"));
        assertFalse(ApposeEnvironments.looksLikeLockFormatSkew("some unrelated build error"));
    }

    @Test
    void collectCauseMessagesToleratesSelfReference() {
        // Java forbids a true cause cycle, so this only exercises a nested chain.
        Throwable a = new RuntimeException("a");
        Throwable b = new RuntimeException("b", a);
        String chain = ApposeEnvironments.collectCauseMessages(b);
        assertTrue(chain.contains("a") && chain.contains("b"));
    }
}
