package uk.ac.soton.ecs;

import ch.akuhn.matrix.Vector;
import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.UniformSampler;
import org.openimaj.util.pair.IntFloatPair;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class Run2 {

    private final static int PATCH_STEP = 4;
    private final static int PATCH_DIM = 8;
    private final static int CLUSTERS = 600;
    private final static int TRAIN_SPLIT = 80;
    private final static int EVAL_SPLIT = 20;

    public static void main(String[] args) throws IOException {

    	Path trainingDataPath = Paths.get("resources/training/");
        Path testingDataPath = Paths.get("resources/testing/");

        // Training and testing data
        VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<>(trainingDataPath.toAbsolutePath().toString(), ImageUtilities.FIMAGE_READER);
        VFSListDataset<FImage> testingData = new VFSListDataset<>(testingDataPath.toAbsolutePath().toString(), ImageUtilities.FIMAGE_READER);
        
        runAlgorithm(trainingData,testingData);
        System.out.println("Finished running algorithm.");
    }

    /**
     * Trains the quantiser, extracts the features, classifies the images and prints the results.
     * @param trainingData The data used for training the quantiser.
     * @param testingData Data used for testing the accuracy.
     */
    private static void runAlgorithm(VFSGroupDataset<FImage> trainingData, VFSListDataset<FImage> testingData) throws IOException {

        // Split training data into two subsets at a TRAIN_SPLIT:EVAL_SPLIT ratio for each class
    	GroupedRandomSplitter<String, FImage> splits =
                new GroupedRandomSplitter<>(trainingData, TRAIN_SPLIT, 0, EVAL_SPLIT);
    	
        // Create a print writer to output the data to a file.
        File outputFile = new File("resources/results/run2.txt");
        PrintWriter writer = new PrintWriter(new FileWriter(outputFile));
		
        HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(splits.getTrainingDataset(), TRAIN_SPLIT*15));
        
        FeatureExtractor<DoubleFV, FImage> extractor = new PatchVectorFeatureExtractor(assigner);

        LiblinearAnnotator<FImage, String> classifier = new LiblinearAnnotator<>(extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        classifier.train(splits.getTrainingDataset());

        // Evalutating precision using the testing subset of the training data.
        ClassificationEvaluator<CMResult<String>, String, FImage> eval =
                new ClassificationEvaluator<>(classifier, splits.getTestDataset(), new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));

        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);
        System.out.println(result.getDetailReport());

        // Writing prediction results to run2.txt.
        int testImageCounter = 0;
        for(FImage img : testingData) {

            List<ScoredAnnotation<String>> imgClass = classifier.annotate(img);
            String predictedClass = imgClass.toString();

            predictedClass = predictedClass.substring(predictedClass.indexOf("(") + 1);
            predictedClass = predictedClass.substring(0, predictedClass.indexOf(","));

            writer.println(testImageCounter + ".jpg " + predictedClass);
            testImageCounter++;
        }

        writer.close();
    }

    /**
     * Builds a HardAssigner from s clustered sample of patches using K-Means clustering.
     * @param groupedDataset The data to build on.
     * @return HardAssigner that can assign features to identifiers.
     */
 	private static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>, FImage> groupedDataset){

 		List<LocalFeatureList<FloatKeypoint>> sampleKeys = new ArrayList<>();

 		System.out.println("Image classes:");
 		System.out.println(groupedDataset.getGroups().toString());

 		int c = 0;
        UniformSampler selector = new UniformSampler<Rectangle>();

 		for (FImage img : groupedDataset) {

            LocalFeatureList<FloatKeypoint> keys = getFeatures(img, PATCH_STEP, PATCH_DIM);

            // Select a quarter of the patch vectors for each image
            selector.setCollection(keys);
            LocalFeatureList<FloatKeypoint> sample = new MemoryLocalFeatureList<>(selector.sample((int) (keys.size() * 0.25)));
            sampleKeys.add(sample);
 		}

 		// K-Means clustering of features
 		FloatKMeans km = FloatKMeans.createKDTreeEnsemble(CLUSTERS);
 		DataSource<float[]> datasource = new LocalFeatureListDataSource<>(sampleKeys);
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
    private static LocalFeatureList<FloatKeypoint> getFeatures(FImage img, int patchStep, int patchDim){

        List<FloatKeypoint> features = new ArrayList<>();

        RectangleSampler sampler = new RectangleSampler(img.normalise(), patchStep, patchStep, patchDim, patchDim);

        for(Rectangle r : sampler.allRectangles()){

            // Extract an image sample, which is flattened into a vector as well as centered and normalised.
            double[] doubleVector = centerAndNormalize(img.normalise().extractROI(r).getDoublePixelVector());

            float[] vector = new float[doubleVector.length];

            // Converting double vector back into a float vector.
            for(int i = 0; i < vector.length; i++){
                vector[i] = (float) doubleVector[i];
            }

            features.add(new FloatKeypoint(r.x, r.y, 0, 1, vector));
        }

        return new MemoryLocalFeatureList<>(features);
    }

    /**
     * A static method which mean-centers and normalizes a vector.
     * @param vectorArray The vector array to be mean-centered and normalized
     * @return The resultant n-dimensional vector in the form of a double[].
     */
    private static double[] centerAndNormalize(double[] vectorArray) {
        Vector vector = Vector.wrap(vectorArray);
        vector.applyCentering();
        vector.times(1/vector.norm());
        return vector.unwrap();
    }

    /**
     * Custom feature extractor implementation of the FeatureExtractor interface.
     */
    static class PatchVectorFeatureExtractor implements FeatureExtractor<DoubleFV, FImage> {

        HardAssigner<float[], float[], IntFloatPair> assigner;

        PatchVectorFeatureExtractor(HardAssigner<float[], float[], IntFloatPair> assigner){
            super();
            this.assigner = assigner;
        }

		@Override
		public DoubleFV extractFeature(FImage img) {

            BagOfVisualWords<float[]> bovw = new BagOfVisualWords<>(assigner);

            // Split image into 4 both vertically and horizontally
            BlockSpatialAggregator<float[], SparseIntFV> spatial = new BlockSpatialAggregator<>(bovw, 4, 4);

            // Normalise and append the computed histograms
            return spatial.aggregate(getFeatures(img, PATCH_STEP, PATCH_DIM), img.getBounds()).normaliseFV();
		}
    }
}
