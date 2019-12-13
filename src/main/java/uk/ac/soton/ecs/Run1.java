package uk.ac.soton.ecs;

import ch.akuhn.matrix.Vector;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.DoubleNearestNeighboursExact;
import org.openimaj.util.pair.IntDoublePair;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class Run1 {
    public static void main(String[] args) throws IOException {
        Path trainingDataPath = Paths.get("resources/training/");
        Path testingDataPath = Paths.get("resources/testing/");
        VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<>(trainingDataPath.toAbsolutePath().toString(), ImageUtilities.FIMAGE_READER);
        VFSListDataset<FImage> testingData = new VFSListDataset<>(testingDataPath.toAbsolutePath().toString(), ImageUtilities.FIMAGE_READER);
        Integer k = 21; //current optimum.

        Map<String,String> classificationMap = runAlgorithm(trainingData,testingData, k);
        writeResultsToFile(classificationMap);
        Double accuracy = Utils.computeAccuracy(Paths.get("resources/results/correct.txt"), Paths.get("resources/results/run1.txt"));
        System.out.println("Accuracy= " + accuracy);
    }

    /**
     * A method which runs a simple Scene Classification algorithm. The algorithm resizes every training image to a 16x16 image, transforms this
     * into a vector and then mean-centers and normalizes it. The same operation is done for all testing images, who's classification is determined by
     * a weighted-vote of it's k-nearest-neighbours.
     * @param trainingData The data which should be used for training the algorithm. It's assumed this is already classified.
     * @param testingData The data which should be used to test the algorithm.
     * @param k The value to be used as k in the k-nearest-neighbours algorithm.
     * @throws IOException If writing to the file fails.
     */
    public static Map<String,String> runAlgorithm(VFSGroupDataset<FImage> trainingData, VFSListDataset<FImage> testingData, Integer k) throws IOException {
        Map<String,String> classificationMap = new TreeMap<>();

        //Create a map to store the 'tiny image' vectors and their respective categories.
        Map<double[],String> vectorMap = new HashMap<>();
        ResizeProcessor resizeProcessor = new ResizeProcessor(16,16,false);

        //Loop over each image in each directory and add their vector representation to the vectorMap.
        trainingData.forEach((directory, imageDataset) -> imageDataset.forEach(trainingImage -> {
            double[] trainingVector = resizeAndGetVector(resizeProcessor, trainingImage);
            vectorMap.put(trainingVector,directory);
        }));

        //Create a DoubleNearestNeighbourExact object using the 256-dimensional vectors computed from the training data.
        double[][] vectors = vectorMap.keySet().toArray(new double[0][0]);
        DoubleNearestNeighboursExact knn = new DoubleNearestNeighboursExact(vectors);

        //Iterate over each image in the testing dataset.
        for(int testImageCounter = 0; testImageCounter < testingData.size(); testImageCounter++ ){
            FImage testImage = testingData.get(testImageCounter);

            //Resize the test image and flatten it to a 256-dimensional vector.
            double[] testVector = resizeAndGetVector(resizeProcessor, testImage);

            //Create a Map which will be used to store the weighted vote for the classification of the point.
            Map<String, Double> voteMap = new HashMap<>();
            List<IntDoublePair> neighbours = knn.searchKNN(testVector, k);

            //Iterate over every neighbour and add its vote to the VoteMap. An inverse distance function is used as the weight function.
            for(IntDoublePair indexAndDistance : neighbours){
                String neighbourClassification = vectorMap.get(vectors[indexAndDistance.first]);
                Double weightedVote = 1 / indexAndDistance.second;
                voteMap.put(neighbourClassification, voteMap.getOrDefault(neighbourClassification, 0.0) + weightedVote);
            }

            //Iterate over all votes in the voteMap. and write to the file the highest-voted classification.
            Double maxDistance = Collections.max(voteMap.values());
            for (String classification: voteMap.keySet()) {
                if (voteMap.get(classification).equals(maxDistance)){
                    classificationMap.put(testImageCounter+".jpg",classification);
                }
            }
        }

        return classificationMap;
    }

    private static boolean writeResultsToFile(Map<String,String> classificationMap) {
        try{
            File outputFile = new File("resources/results/run1.txt");
            PrintWriter writer = new PrintWriter(new FileWriter(outputFile));

            for(String mapKey : classificationMap.keySet()){
                writer.write(mapKey + " " + classificationMap.get(mapKey));
            }
            writer.close();
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    /**
     * A static method which resize an image using a given FImage and ResizeProcessor, then mean-centers the vector, normalizes the vector, and returns the result.
     * @param resizeProcessor The ResizeProcessor which should be used when resizing the image (Recommended: 16x16 without aspect-ratio preservation).
     * @param image The image from which the resultant vector should be based on.
     * @return The resultant double[] representing an n-dimensional vector.
     */
    private static double[] resizeAndGetVector(ResizeProcessor resizeProcessor, FImage image){
        resizeProcessor.processImage(image);
        return centerAndNormalize(image.getDoublePixelVector());
    }

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
