package uk.ac.soton.ecs;

import ch.akuhn.matrix.Vector;
import org.apache.avro.generic.GenericData;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
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



        for(FImage img : trainingData){
            List<double[]> patchVectors = getPatchVectors(img, 4, 8);
        }

        FloatKMeans cluster = FloatKMeans.createExact(500);

        writer.close();
    }

    /**
     * Extracts patches from the image at a given step and flattens them into a list of vectors.
     * @param img Image to be sampled.
     * @param patchStep Step in x and y direction for sampling.
     * @param patchDim Sample dimension (patch size = patchDim x patchDim).
     * @return List of sampled patches flattened into vectors.
     */
    private static List<double[]> getPatchVectors(FImage img, int patchStep, int patchDim){

        List<double[]> vectors = new ArrayList<>();

        RectangleSampler sampler = new RectangleSampler(img.normalise(), patchStep, patchStep, patchDim, patchDim);

        for(Rectangle r : sampler.allRectangles()){

            double[] vector = img.normalise().extractROI(r).getDoublePixelVector();
            vectors.add(vector);
        }

        return vectors;
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
}
