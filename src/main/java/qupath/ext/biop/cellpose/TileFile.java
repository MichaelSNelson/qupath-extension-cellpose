/*-
 * Copyright 2020-2021 BioImaging & Optics Platform BIOP, Ecole Polytechnique Fédérale de Lausanne
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

import org.apache.commons.io.FilenameUtils;
import qupath.lib.objects.PathObject;
import qupath.lib.regions.RegionRequest;

import java.io.File;
import java.util.Collection;
import java.util.Collections;

/**
 * Holds the correspondence between a {@link RegionRequest} and a saved input tile image on disk,
 * and infers the resulting Cellpose label ("mask") file name.
 * <p>
 * This used to be a private static inner class of {@link Cellpose2D}. It was promoted to a
 * package-visible top-level type so that the pluggable segmentation backends in
 * {@code qupath.ext.biop.cellpose.backend} can share it (the backend package only needs the
 * public accessors below; candidate storage stays package-private to {@code qupath.ext.biop.cellpose}).
 * The behaviour is otherwise unchanged.
 *
 * @author Olivier Burri
 */
public class TileFile {
    private final RegionRequest request;
    private final File imageFile;
    private final PathObject parent;

    private Collection<CandidateObject> candidates = Collections.emptyList();

    TileFile(RegionRequest request, File imageFile, PathObject parent) {
        this.request = request;
        this.parent = parent;
        this.imageFile = imageFile;
    }

    /**
     * @return the saved input tile image on disk
     */
    public File getImageFile() {
        return imageFile;
    }

    /**
     * @return the Cellpose label file that is expected to be produced for this tile
     * (the input file name with the {@code _cp_masks.tif} suffix)
     */
    public File getLabelFile() {
        return new File(FilenameUtils.removeExtension(imageFile.getAbsolutePath()) + "_cp_masks.tif");
    }

    /**
     * @return the region request describing the location of this tile in the source image
     */
    public RegionRequest getTile() {
        return request;
    }

    /**
     * @return the real parent object of this tile
     */
    public PathObject getParent() {
        return parent;
    }

    Collection<CandidateObject> getCandidates() {
        return this.candidates;
    }

    void setCandidates(Collection<CandidateObject> candidates) {
        this.candidates = candidates;
    }
}
