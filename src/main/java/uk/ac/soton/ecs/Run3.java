package uk.ac.soton.ecs;

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
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;


public class Run3 {

    final static int MAX_FEATURES = 10000;
    final static int CLUSTERS = 500;

    public static void main(String[] args) throws IOException {
        Path trainingDataPath = Paths.get("resources/training/");
        Path testingDataPath = Paths.get("resources/testing/");
        VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<>(trainingDataPath.toAbsolutePath().toString(), ImageUtilities.FIMAGE_READER);
        VFSListDataset<FImage> testingData = new VFSListDataset<>(testingDataPath.toAbsolutePath().toString(), ImageUtilities.FIMAGE_READER);

        //Split training data into training and testing subsets
        GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<>(trainingData, 80, 0, 20);
        GroupedDataset<String, ListDataset<FImage>, FImage> trainingSplit = splits.getTrainingDataset();
        GroupedDataset<String, ListDataset<FImage>, FImage> testingSplit = splits.getTestDataset();

        DenseSIFT dsift = new DenseSIFT(5, 7);
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<>(dsift, 6f, 7);

        HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(trainingSplit, 100), pdsift);

        FeatureExtractor<SparseIntFV, FImage> extractor = new SIFTExtractor(pdsift, assigner);

        //Construct and train linear classifier that uses a homogeneous kernel map
        HomogeneousKernelMap homogeneousKernelMap = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
        LiblinearAnnotator<FImage, String> classifier = new LiblinearAnnotator<>(homogeneousKernelMap.createWrappedExtractor(extractor), LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        classifier.train(trainingSplit);
        System.out.println("Trained the classifier");

        //Evaluate classifier's accuracy on testing subset
        ClassificationEvaluator<CMResult<String>, String, FImage> eval = new ClassificationEvaluator<>(classifier, testingSplit, new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));
        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);
        System.out.println(result.getDetailReport());
        System.out.println("Evaluated accuracy on testing subset from training data");

        //Predict classes for images in the testing data
        makeClassPredictions(classifier, testingData);
        System.out.println("Completed predictions on testing data");
    }

    /**
     * Extracts SIFT features from images, and then clusters them into 500 separate classes
     *
     * @param trainingSplit The images
     * @param pdsift        The extractor
     * @return The assigner used to assign SIFT features to identifiers
     */
    static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSplit, PyramidDenseSIFT<FImage> pdsift) {
        List<LocalFeatureList<ByteDSIFTKeypoint>> features = new ArrayList<>();

        for (String directory : trainingSplit.getGroups()) {
            ListDataset<FImage> directoryImages = trainingSplit.getInstances(directory);
            for (FImage image : directoryImages) {
                pdsift.analyseImage(image);
                features.add(pdsift.getByteKeypoints(0.005f));
            }
        }

        if (features.size() > MAX_FEATURES) {
            features = features.subList(0, MAX_FEATURES);
        }

        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(CLUSTERS);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<>(features);
        ByteCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }

    /**
     * FeatureExtractor implementation with which we train the classifier
     */
    static class SIFTExtractor implements FeatureExtractor<SparseIntFV, FImage> {
        PyramidDenseSIFT<FImage> pdsift;
        HardAssigner<byte[], float[], IntFloatPair> assigner;

        public SIFTExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner) {
            this.pdsift = pdsift;
            this.assigner = assigner;
        }

        public SparseIntFV extractFeature(FImage image) {
            pdsift.analyseImage(image);
            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);
            return bovw.aggregate(pdsift.getByteKeypoints(0.015f));
        }
    }

    /**
     * Predicts the class of each image in the dataset and writes the predictions to a file
     *
     * @param classifier  The classifier used for predictions
     * @param testingData The dataset of images
     * @throws IOException
     */
    static void makeClassPredictions(LiblinearAnnotator<FImage, String> classifier, VFSListDataset<FImage> testingData) throws IOException {
        File outputFile = new File("resources/results/run3.txt");
        PrintWriter writer = new PrintWriter(new FileWriter(outputFile));

        int testImageCounter = 0;
        for (FImage image : testingData) {
            List<ScoredAnnotation<String>> imageClass = classifier.annotate(image);
            String predictedClass = imageClass.toString();

            predictedClass = predictedClass.substring(predictedClass.indexOf("(") + 1);
            predictedClass = predictedClass.substring(0, predictedClass.indexOf(","));

            writer.println(testImageCounter + ".jpg " + predictedClass);
            testImageCounter++;
        }

        writer.close();
    }


}
