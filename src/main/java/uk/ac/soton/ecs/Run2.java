package uk.ac.soton.ecs;

import ch.akuhn.matrix.Vector;
import org.apache.commons.vfs2.FileObject;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.Image;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class Run2 {
    public static void main(String[] args) throws IOException {
        Path trainingDataPath = Paths.get("resources/training/");
        Path testingDataPath = Paths.get("resources/testing/");
        VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<>(trainingDataPath.toAbsolutePath().toString(), ImageUtilities.FIMAGE_READER);
        VFSListDataset<FImage> testingData = new VFSListDataset<>(testingDataPath.toAbsolutePath().toString(), ImageUtilities.FIMAGE_READER);

        runAlgorithm(trainingData,testingData);
        Double accuracy = Utils.computeAccuracy(Paths.get("resources/results/correct.txt"), Paths.get("resources/results/run1.txt"));
        System.out.println("Accuracy = " + accuracy);
    }

    public static void runAlgorithm(VFSGroupDataset<FImage> trainingData, VFSListDataset<FImage> testingData) throws IOException {

        //Create a print writer to output the data to a file.
        File outputFile = new File("resources/results/run2.txt");
        PrintWriter writer = new PrintWriter(new FileWriter(outputFile));

        HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(trainingData);
        
        for(FImage img : trainingData){

            

            List<FloatKeypoint> features = getFeatures(img, 4, 8);
        }

        FloatKMeans cluster = FloatKMeans.createExact(500);

        writer.close();
    }

    // Extracts the first 10000 dense SIFT features from the images in the given dataset
 	static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(
 			VFSGroupDataset<FImage> groupedDataset){

 		List<LocalFeatureList<FloatKeypoint>> allkeys = new ArrayList<>();

 		// Record the list of features extracted from each image
 		for (FImage rec: groupedDataset) {
 			allkeys.add((LocalFeatureList<FloatKeypoint>) getFeatures(rec.getImage(),4 ,8));
 		}
 		
 		if (allkeys.size() > (int) allkeys.size()*0.2)
 			allkeys = allkeys.subList(0, (int) (allkeys.size()*0.2));

 		// Cluster sample of features using K-Means
 		FloatKMeans km = FloatKMeans.createKDTreeEnsemble(600);
 		DataSource<float[]> datasource = new LocalFeatureListDataSource<FloatKeypoint, float[]>(allkeys);
 		FloatCentroidsResult result = km.cluster(datasource);

 		return result.defaultHardAssigner();
 		
 	}
 	

    
    /**
     * Extracts patches from the image at a given step and flattens them into a list of feature vectors.
     * @param img Image to be sampled.
     * @param patchStep Step in x and y direction for sampling.
     * @param patchDim Sample dimension (patch size = patchDim x patchDim).
     * @return List of sampled patches flattened into feature vectors.
     */
    private static List<FloatKeypoint> getFeatures(FImage img, int patchStep, int patchDim){

        List<FloatKeypoint> features = new ArrayList<>();

        RectangleSampler sampler = new RectangleSampler(img.normalise(), patchStep, patchStep, patchDim, patchDim);

        for(Rectangle r : sampler.allRectangles()){

            float[] vector = img.normalise().extractROI(r).getFloatPixelVector();

            // Do I need to center and normalize or does this already do the trick?
            features.add(new FloatKeypoint(r.x, r.y, 0, 1, vector));
        }

        return features;
    }

    /**
     * Samples a list of 8x8 patches for an image every 4 pixels in the x and y direction.
     * @param img The image to be sampled.
     * @return List of 8x8 patches.
     */
//    private static List<float[][]> getPatches(FImage img){
//
//        float[][] pixels = img.pixels;
//        List<float[][]> patches = new ArrayList<>();
//
//        // Sample every 4 pixels
//        for(int r = 0; r < pixels.length; r += 4){
//            for(int c = 0; c < pixels[c].length; c += 4){
//
//                float[][] patch = new float[8][8];
//
//                // Populate patch with pixels
//                for(int i = 0; i < patch.length; i++){
//                    for(int j = 0; j < patch[i].length; j++){
//
//                        if(r+i < pixels.length && c+j < pixels[i].length){
//                            patch[i][j] = pixels[r+i][c+j];
//                        }
//                    }
//                }
//            }
//        }
//
//        return patches;
//    }

//    private static double[] flattenAndAdjust(float[][] patch){
//
//        double flat[] = new double[patch.length * patch[0].length];
//        int idx = 0;
//
//        for (int i = 0; i < patch.length; i++){
//
//            for (int j = 0; j < patch.length; j++){
//                flat[idx++] = patch[i][j];
//            }
//        }
//
//        return centerAndNormalize(flat);
//    }

    /**
     * A static method which mean-centers and normalizes a vector.
     * @param vectorArray The vector array to be mean-centered and normalized
     * @return The resultant n-dimensional vector in the form of a double[].
     */
    private static double[] centerAndNormalize(double[] vectorArray){
        Vector vector = Vector.wrap(vectorArray);
        vector.applyCentering();
        vector.times(1/vector.norm());
        return vector.unwrap();
    }

    class PatchVectorFeatureExtractor implements FeatureExtractor<DoubleFV, Caltech101.Record<FImage>> {

        HardAssigner<float[], float[], IntFloatPair> assigner;

        public PatchVectorFeatureExtractor(HardAssigner<float[], float[], IntFloatPair> assigner){

            this.assigner = assigner;
        }

        public DoubleFV extractFeature(Caltech101.Record<FImage> r) {

            FImage img = r.getImage();

            BagOfVisualWords<float[]> bovw = new BagOfVisualWords<>(assigner);

            BlockSpatialAggregator<float[], SparseIntFV> spatial =
                    new BlockSpatialAggregator<>(bovw, 2, 2);

            // Append and normalise the resultant spatial histograms
            return spatial.aggregate(getFeatures(img, 4, 8), img.getBounds()).normaliseFV();

        }
    }
}
