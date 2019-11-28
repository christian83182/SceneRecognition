package uk.ac.soton.ecs;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.DoubleNearestNeighboursExact;
import org.openimaj.util.pair.IntDoublePair;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Run1 {
    public static void main(String[] args) throws IOException {

        //Specify the paths for the training and testing data. Useful since location will be relative.
        Path trainingDataPath = Paths.get("resources/training/").toAbsolutePath();
        Path testingDataPath = Paths.get("resources/testing/").toAbsolutePath();

        //Create datasets for the training and testing data using their respective paths.
        VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<>(trainingDataPath.toString(), ImageUtilities.FIMAGE_READER);
        VFSListDataset<FImage> testingData = new VFSListDataset<>(testingDataPath.toString(), ImageUtilities.FIMAGE_READER);

        //Create a print writer to output the data to a file.
        PrintWriter writer = new PrintWriter(new FileWriter("run1.txt"));

        //Create a map to store the 'tiny image' vectors and their respective categories.
        Map<double[],String> vectorMap = new HashMap<>();

        //The resize processor is used to resize images to a 16x16 area. Aspect ratio is not maintained.
        ResizeProcessor resizeProcessor = new ResizeProcessor(16,16,false);

        //Iterate over each image in each directory.
        trainingData.forEach((directory, imageDataset) -> imageDataset.forEach(trainingImage -> {
            //Resize the image and add it's 256-dimensional vector to the map, pointing to its respective classification.
            resizeProcessor.processImage(trainingImage);
            double[] trainingVector = trainingImage.getDoublePixelVector();
            vectorMap.put(trainingVector,directory);
        }));

        //Create a DoubleNearestNeighbourExact object using the 256-dimensional vectors computed from the training data.
        double[][] vectors = vectorMap.keySet().toArray(new double[0][0]);
        DoubleNearestNeighboursExact knn = new DoubleNearestNeighboursExact(vectors);

        //Iterate over each image in the testing dataset.
        for(int testImageCounter = 0; testImageCounter < testingData.size(); testImageCounter++ ){
            FImage testImage = testingData.get(testImageCounter);

            //Resize the test image and flatten it to a 256-dimensional vector.
            resizeProcessor.processImage(testImage);
            double[] testVector = testImage.getDoublePixelVector();

            //Create a Map which will be used to store the weighted vote for the classification of the point.
            Map<String, Double> voteMap = new HashMap<>();

            //Use the DoubleNearestNeighbourExact object to find the k nearest neighbours.
            List<IntDoublePair> neighbours = knn.searchKNN(testVector, 5);

            //Iterate over evey neighbour.
            for(IntDoublePair indexAndDistance : neighbours){
                //Get the neighbour's classification from it's index in the list of vectors given to knn.
                String neighbourClassification = vectorMap.get(vectors[indexAndDistance.first]);
                //compute the weight using a kernel (reciprocal distance in this case).
                Double weightedVote = 1 / indexAndDistance.second;
                //Add the weightedVote to the relevant classification in voteMap.
                voteMap.put(neighbourClassification, voteMap.getOrDefault(neighbourClassification, 0.0) + weightedVote);
            }

            //use Collections.max() to find the highest voted classification
            Double maxDistance = Collections.max(voteMap.values());

            //Iterate over all votes in the voteMap.
            for (String classification: voteMap.keySet()) {
                if (voteMap.get(classification).equals(maxDistance)){
                    writer.println(testImageCounter +".jpg " + classification);
                }
            }
        }

        //Close the writer used to output to the file.
        writer.close();
    }
}
