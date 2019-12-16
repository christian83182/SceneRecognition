package uk.ac.soton.ecs;

import ch.akuhn.matrix.Vector;
import com.hp.hpl.jena.util.cache.RandCache;
import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.util.UniformSampler;
import uk.ac.soton.ecs.Utils;

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
import org.openimaj.image.DisplayUtilities;
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
import org.openimaj.util.pair.IntFloatPair;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class Run2 {
    public static void main(String[] args) throws IOException {
        
    	
    	Path trainingDataPath = Paths.get("resources/training/");
        Path testingDataPath = Paths.get("resources/testing/");

        // Training and testing data
        VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<>(trainingDataPath.toAbsolutePath().toString(), ImageUtilities.FIMAGE_READER);
        VFSListDataset<FImage> testingData = new VFSListDataset<>(testingDataPath.toAbsolutePath().toString(), ImageUtilities.FIMAGE_READER);
        
        runAlgorithm(trainingData,testingData);

        Double accuracy = Utils.computeAccuracy(Paths.get("resources/results/correct.txt"), Paths.get("resources/results/run2.txt"));
        System.out.println("Accuracy= " + accuracy);
        System.out.println("Fininshed runnning the algorithm");
    }

    /**
     * Trains the quantiser, extracts the features, classifies the images and prints the results.
     * @param trainingData The data used for training the quantiser.
     * @param testingData Data used for testing the accuracy.
     */
    public static void runAlgorithm(VFSGroupDataset<FImage> trainingData, VFSListDataset<FImage> testingData) throws IOException {
    	
    	
    	GroupedRandomSplitter<String, FImage> splits =
                new GroupedRandomSplitter<>(trainingData, 10, 0, 90);

    	
        //Create a print writer to output the data to a file.
        File outputFile = new File("resources/results/run2.txt");
        PrintWriter writer = new PrintWriter(new FileWriter(outputFile));
		BufferedWriter bw = new BufferedWriter(writer);
        
		
        HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(splits.getTrainingDataset(), 150));
        
        FeatureExtractor<DoubleFV, FImage> extractor = new PatchVectorFeatureExtractor(assigner);

        LiblinearAnnotator<FImage, String> classifier = new LiblinearAnnotator<>(extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        classifier.train(splits.getTrainingDataset());
        
        ClassificationEvaluator<CMResult<String>, String, FImage> eval =
                new ClassificationEvaluator<>(classifier, splits.getTestDataset(), new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));

        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);
        System.out.println(result.getDetailReport());
        
     		/*try {
     			
     			if (!outputFile.exists()) {
     				outputFile.createNewFile();
     			}
     			else {
     				
     				int testImageCounter = 0;
     				for(FImage img : testingData) {
     					
     					List<ScoredAnnotation<String>> imgClass = classifier.annotate(img);
     					String predictedClass = imgClass.toString();
     					
     					predictedClass = predictedClass.substring(predictedClass.indexOf("(") + 1);
     					predictedClass = predictedClass.substring(0, predictedClass.indexOf(","));
     					
     					writer.println(testImageCounter + ".jpg " + predictedClass);
     					testImageCounter++;
     				}
     			}

     		} catch (IOException e) {
     			e.printStackTrace();
     		}*/

        writer.close();
    }

 	static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>, FImage> groupedDataset){

 		List<LocalFeatureList<FloatKeypoint>> sampleKeys = new ArrayList<>();

 		System.out.println(groupedDataset.getGroups().toString());
 		
 		// Record the list of features extracted from each image
 		int c = 0;
 		for (FImage img : groupedDataset) {

            LocalFeatureList<FloatKeypoint> keys = getFeatures(img, 4, 8);
 			
 			System.out.println(++c);

            int size = keys.size();
            Random rand = new Random();

            for (int i = size - 1; i >= (size - 0.2*size); --i) {
                Collections.swap(keys, i, rand.nextInt(i + 1));
            }

 			sampleKeys.add(keys.subList((int) (size - 0.2*size), size));
 		}
 		
 		/*if (allkeys.size() > allkeys.size() * 0.5) {
 			
 			allkeys = allkeys.subList(0, (int) (allkeys.size() * 0.5));
 		}*/

 		// Cluster sample of features using K-Means
 		FloatKMeans km = FloatKMeans.createKDTreeEnsemble(500);

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
//        UniformSampler selector = new UniformSampler<Rectangle>();
//        selector.setCollection(sampler.allRectangles());
//        List<Rectangle> samples = selector.sample((int) (sampler.allRectangles().size() * 0.5));

        for(Rectangle r : sampler.allRectangles()){

            float[] vector = img.normalise().extractROI(r).getFloatPixelVector();

            // Do I need to center and normalize or does this already do the trick?
            features.add(new FloatKeypoint(r.x, r.y, 0, 1, vector));
        }


        MemoryLocalFeatureList<FloatKeypoint> featureList = new MemoryLocalFeatureList<>(features);
        
        return featureList;
    }
    
    static class PatchVectorFeatureExtractor implements FeatureExtractor<DoubleFV, FImage> {

        HardAssigner<float[], float[], IntFloatPair> assigner;

        public PatchVectorFeatureExtractor(HardAssigner<float[], float[], IntFloatPair> assigner){
            super();
            this.assigner = assigner;
        }

		@Override
		public DoubleFV extractFeature(FImage img) {

            BagOfVisualWords<float[]> bovw = new BagOfVisualWords<>(assigner);

            BlockSpatialAggregator<float[], SparseIntFV> spatial = new BlockSpatialAggregator<>(bovw, 4, 4);

            // Append and normalise the resultant spatial histograms
            return spatial.aggregate(getFeatures(img, 4, 8), img.getBounds()).normaliseFV();
		}
    }
}
