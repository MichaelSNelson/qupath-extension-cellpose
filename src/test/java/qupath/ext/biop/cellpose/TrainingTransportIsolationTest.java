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

package qupath.ext.biop.cellpose;

import org.junit.jupiter.api.Test;
import qupath.ext.biop.cellpose.backend.CellposeBackend;
import qupath.ext.biop.cellpose.backend.CellposeTransport;
import qupath.ext.biop.cellpose.backend.SubprocessBackend;

import java.util.LinkedHashMap;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

/**
 * Regression guard for a design invariant: the training and QC (validation-image) paths always run
 * through the external subprocess transport, independent of the detection transport preference or
 * builder flag. The in-process (Appose) backend implements inference only; it has no training or
 * validation entry point, so selecting {@link CellposeTransport#APPOSE} for detection must never
 * route training or its QC through Appose.
 * <p>
 * This asserts the code-level isolation only. Custom-model <em>training</em> itself is not exercised
 * here (it requires a configured Cellpose training environment and real ground-truth data) and has
 * not been tested against this change end-to-end -- see the README/NOTICE caveat.
 */
class TrainingTransportIsolationTest {

    /** Minimal instance carrying just the fields the training-support backend reads. */
    private static Cellpose2D instanceWithTransport(CellposeTransport transport) {
        Cellpose2D cellpose = new Cellpose2D();
        cellpose.transport = transport;
        cellpose.parameters = new LinkedHashMap<>();
        cellpose.model = "cyto3";
        cellpose.disableGPU = false;
        cellpose.doReadResultsAsynchronously = false;
        return cellpose;
    }

    @Test
    void trainingSupportStaysSubprocessWhenApposeSelected() {
        Cellpose2D cellpose = instanceWithTransport(CellposeTransport.APPOSE);
        // Precondition: this instance really does request the in-process (Appose) detection transport.
        assertEquals(CellposeTransport.APPOSE, cellpose.transport);
        CellposeBackend backend = cellpose.createTrainingSupportBackend();
        assertInstanceOf(SubprocessBackend.class, backend,
                "training/QC must always run through the subprocess transport, never Appose");
    }

    @Test
    void trainingSupportIsSubprocessForEveryTransportSetting() {
        CellposeTransport[] settings = {CellposeTransport.SUBPROCESS, CellposeTransport.APPOSE, null};
        for (CellposeTransport t : settings) {
            CellposeBackend backend = instanceWithTransport(t).createTrainingSupportBackend();
            assertInstanceOf(SubprocessBackend.class, backend,
                    "training-support transport must not depend on the detection transport (" + t + ")");
        }
    }
}
