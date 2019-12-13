package uk.ac.soton.ecs;

import ch.akuhn.matrix.Vector;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.dataset.util.DatasetAdaptors;
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
import java.util.List;

public class Run1 {
    public static void main(String[] args) throws IOException, InterruptedException {
        Path trainingDataPath = Paths.get("resources/training/");
        Path testingDataPath = Paths.get("resources/testing/");

        VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<>(trainingDataPath.toAbsolutePath().toString(), ImageUtilities.FIMAGE_READER);
        VFSListDataset<FImage> testingData = new VFSListDataset<>(testingDataPath.toAbsolutePath().toString(), ImageUtilities.FIMAGE_READER);

        int k = findOptimum(convertToGroupedDataset(trainingData),1,100);
        System.out.println("\n--------------- DONE -----------------");
        System.out.println("Optimum K: " + k);
        System.out.println("----------------------------------------\n");

        Map<String,String> classificationMap = runAlgorithm(convertToGroupedDataset(trainingData),testingData, k);
        writeResultsToFile(classificationMap);
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
    public static Map<String,String> runAlgorithm(GroupedDataset<String, ListDataset<FImage>, FImage> trainingData, ListDataset<FImage> testingData, Integer k) throws IOException {
        Map<String,String> classificationMap = new LinkedHashMap<>();

        //Create a map to store the 'tiny image' vectors and their respective categories.
        Map<double[],String> vectorMap = new HashMap<>();

        //Loop over each image in each directory and add their vector representation to the vectorMap.
        System.out.println("[INFO] Computing 'tiny-image' vector for each image... ");
        trainingData.forEach((directory, imageDataset) -> imageDataset.forEach(trainingImage -> {
            double[] trainingVector = resizeAndGetVector(trainingImage);
            vectorMap.put(trainingVector,directory);
        }));

        //Create a DoubleNearestNeighbourExact object using the 256-dimensional vectors computed from the training data.
        System.out.println("[INFO] Adding vectors to vector space...");
        double[][] vectors = vectorMap.keySet().toArray(new double[0][0]);
        DoubleNearestNeighboursExact knn = new DoubleNearestNeighboursExact(vectors);

        //Iterate over each image in the testing dataset.
        for(int testImageCounter = 0; testImageCounter < testingData.size(); testImageCounter++ ){
            if(testImageCounter %250 ==0){
                System.out.println("[INFO] Classifying testing images " + testImageCounter + "/" + testingData.size());
            }
            FImage testImage = testingData.get(testImageCounter);

            //Resize the test image and flatten it to a 256-dimensional vector.
            double[] testVector = resizeAndGetVector(testImage);

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

        System.out.println("[INFO] Done");
        return classificationMap;
    }

    private static boolean writeResultsToFile(Map<String,String> classificationMap) {
        try{
            File outputFile = new File("resources/results/run1.txt");
            PrintWriter writer = new PrintWriter(new FileWriter(outputFile));

            for(String mapKey : classificationMap.keySet()){
                writer.write(mapKey + " " + classificationMap.get(mapKey) +"\n");
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
     * @param image The image from which the resultant vector should be based on.
     * @return The resultant double[] representing an n-dimensional vector.
     */
    private static double[] resizeAndGetVector(FImage image){
        int minDim = Math.min(image.height, image.width);
        image = image.extractCenter(minDim, minDim);
        ResizeProcessor.zoomInplace(image,16,16);
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

    private static GroupedDataset<String, ListDataset<FImage>, FImage> convertToGroupedDataset(VFSGroupDataset<FImage> dataset){
        GroupedDataset<String, ListDataset<FImage>, FImage> newDataset = new MapBackedDataset<>();
        dataset.forEach(newDataset::put);
        return newDataset;
    }

    private static int findOptimum(GroupedDataset<String, ListDataset<FImage>, FImage> dataset, int upperBound, int lowerBound) throws InterruptedException, IOException {
        final Double GOLDEN_RATIO = (Math.sqrt(5)+1)/2;

        int firstSearchPoint =  (int)(lowerBound - ( lowerBound - upperBound ) / GOLDEN_RATIO);
        int secondSearchPoint =  (int)(upperBound + ( lowerBound - upperBound ) / GOLDEN_RATIO);

        while(firstSearchPoint - secondSearchPoint > 1){
            if(computeMeanAccuracy(dataset,firstSearchPoint) < computeMeanAccuracy(dataset,secondSearchPoint)){
                lowerBound = secondSearchPoint;
            } else {
                upperBound = firstSearchPoint;
            }
            firstSearchPoint =  (int)(lowerBound - ( lowerBound - upperBound ) / GOLDEN_RATIO);
            secondSearchPoint = (int)(upperBound + ( lowerBound - upperBound ) / GOLDEN_RATIO);

            System.out.println("[INFO] Current Optimum: "+ ((upperBound+lowerBound)/2));
        }
        return (int)(Math.round((upperBound+lowerBound)/2.0));
    }

    private static double testFunc(double x){
        return Math.pow((x-2),2);
    }

    private static double computeMeanAccuracy(GroupedDataset<String, ListDataset<FImage>, FImage> dataset, int k) throws IOException {
        int repetitions = 10;
        DoubleSummaryStatistics stats = new DoubleSummaryStatistics();

        for(int i = 0; i < repetitions; i++){
            GroupedRandomSplitter<String,FImage> splitter = new GroupedRandomSplitter<>(dataset, 80,0,20);
            GroupedDataset<String, ListDataset<FImage>, FImage> trainingDataset = splitter.getTrainingDataset();
            GroupedDataset<String, ListDataset<FImage>, FImage> testingDataset = splitter.getTestDataset();
            ListDataset<FImage> flatTestingDataset = new ListBackedDataset<>(DatasetAdaptors.asList(testingDataset));

            Map<String,String> resultMap = runAlgorithm(trainingDataset, flatTestingDataset,k);
            HashMap<String,String> correctMap = new LinkedHashMap<>();

            int counter = 0;
            for(String category : testingDataset.keySet()){
                for(FImage ignored : testingDataset.get(category)){
                    correctMap.put(counter + ".jpg", category);
                    counter++;
                }
            }

            Double accuracy = Utils.computeAccuracy(correctMap, resultMap);
            System.out.println("[INFO] Accuracy for run " + (i+1) + "/" + repetitions +" with k=" + k +": " + accuracy);
            stats.accept(accuracy);
        }
        return stats.getAverage();
    }

}
