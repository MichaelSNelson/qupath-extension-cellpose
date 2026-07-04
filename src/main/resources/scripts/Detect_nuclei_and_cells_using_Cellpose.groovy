/* Last tested on QuPath-0.7.0
 * 
 * This scripts requires qupath-extension-cellpose 
 * cf https://github.com/BIOP/qupath-extension-cellpose
 */

// some qp that we need to detect objects and measure them
def imageData  = getCurrentImageData()
def server = getCurrentServer()
def cal = server.getPixelCalibration()
def downsample = 1.0

// if nothing Annotation is selected , let's create a full image annotation
def pathObjects = getSelectedObjects()
if (pathObjects.isEmpty()) {
    createSelectAllObject(true)
}

clearDetections()

// Create a Cellpose detectors for cyto and nuclei
def pathModel_cyto = 'cyto3'
def cellpose_cyto = Cellpose2D.builder( pathModel_cyto )
        .channels( "HCS","DAPI" )
        .pixelSize( 0.3 )              // Resolution for detection
        .diameter(30 )                  // Median object diameter. Set to 0.0 for the `bact_omni` model or for automatic computation
        .measureShape()                // Add shape measurements
        .measureIntensity()            // Add cell measurements (in all compartments) 
        .build()

def pathModel_nuc = 'cyto3'
def cellpose_nuc = Cellpose2D.builder( pathModel_nuc )
        .channels("DAPI" )
        .pixelSize( 0.3 )              // Resolution for detection
        .diameter(10)                  // Median object diameter. Set to 0.0 for the `bact_omni` model or for automatic computation
        .build()

// Run detection for the selected pathObjects and store resulting detections
cellpose_cyto.detectObjects(imageData, pathObjects)
cytos = getDetectionObjects()
cellpose_nuc.detectObjects(imageData, pathObjects)
nucs = getDetectionObjects()
//if one wants to check how each step is doing, uncomment the 4 lines below
//cytos.each{ it.setPathClass(getPathClass("Cyto"))}
//nucs.each{ it.setPathClass(getPathClass("Nuc"))}
//addObjects(cytos) // needed because cellpose detectors remove existing detections
//return

// make sure to clear everything 
clearDetections()

println "Combining ${cytos.size()} cytos and ${nucs.size()} nuclei to create cell objects"
// For combining we simply check that the nuclei center is inside the cell center

// Build a spatial index to efficiently query nearby nuclei.
def tree = new STRtree()
nucs.each { nuc ->
    def roi = nuc.getROI()
    def indexedNucleus = [
        roi: roi,
        x: roi.getCentroidX(),
        y: roi.getCentroidY()
    ]
    def envelope = GeometryTools.roiToEnvelope(roi)
    tree.insert(envelope, indexedNucleus)
}
tree.build()

def cells = []
for (cyto in cytos) {
    def cytoROI = cyto.getROI()
    def envelope = GeometryTools.roiToEnvelope(cytoROI)
    def candidates = tree.query(envelope)

    for (candNuc in candidates) {
        // Keep only nuclei whose centroids lie inside the cytoplasm.
        if (cytoROI.contains(candNuc.x, candNuc.y)) {
            def cellObject = PathObjects.createCellObject(
                cytoROI,
                candNuc.roi,
                getPathClass("Cellpose"),
                null
            )
            cells.add(cellObject)
            break // Cell objects can contain only one nucleus, so stop after the first match.
        }
    }
}
addObjects(cells)

// Intensity & Shape Measurements
// adapted from : https://forum.image.sc/t/transferring-segmentation-predictions-from-custom-masks-to-qupath/43408/12
def measurements = ObjectMeasurements.Measurements.values() as List
def compartments = ObjectMeasurements.Compartments.values() as List // Won't mean much if they aren't cells...
def shape = ObjectMeasurements.ShapeFeatures.values() as List
for (cell in getCellObjects()) {
    ObjectMeasurements.addIntensityMeasurements( server, cell, downsample, measurements, compartments )
    ObjectMeasurements.addCellShapeMeasurements( cell, cal,  shape )
}
fireHierarchyUpdate()
println 'Done!'

/*
 * imports
 */
import qupath.ext.biop.cellpose.Cellpose2D
import qupath.lib.analysis.features.ObjectMeasurements
import qupath.lib.roi.GeometryTools
import org.locationtech.jts.index.strtree.STRtree