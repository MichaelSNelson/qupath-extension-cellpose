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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import qupath.ext.biop.cellpose.CellposeExtension;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * The Appose environment is several gigabytes, so where it lands is a real decision on a shared
 * workstation rather than a detail. These pin the two halves of that: an unset preference must keep
 * the historical default, and a set one must be honoured exactly.
 */
public class EnvironmentLocationTest {

    @AfterEach
    void clearPreference() {
        CellposeExtension.setApposeEnvDirPreference("");
    }

    @Test
    void defaultsToTheApposeLocationWhenUnset() {
        CellposeExtension.setApposeEnvDirPreference("");
        Path expected = Paths.get(System.getProperty("user.home"), ".local", "share", "appose");
        Assertions.assertEquals(expected, ApposeEnvironments.getEnvironmentBase());
    }

    @Test
    void honoursAConfiguredDirectory() {
        CellposeExtension.setApposeEnvDirPreference("/data/shared/appose");
        Assertions.assertEquals(Paths.get("/data/shared/appose"), ApposeEnvironments.getEnvironmentBase());
    }

    @Test
    void surroundingWhitespaceIsIgnored() {
        // Paths get pasted into the preference field; a stray space would otherwise become a
        // directory name that pixi cannot build in.
        CellposeExtension.setApposeEnvDirPreference("  /data/shared/appose  ");
        Assertions.assertEquals(Paths.get("/data/shared/appose"), ApposeEnvironments.getEnvironmentBase());
    }

    @Test
    void blankIsTreatedAsUnset() {
        CellposeExtension.setApposeEnvDirPreference("   ");
        Path expected = Paths.get(System.getProperty("user.home"), ".local", "share", "appose");
        Assertions.assertEquals(expected, ApposeEnvironments.getEnvironmentBase());
    }
}
