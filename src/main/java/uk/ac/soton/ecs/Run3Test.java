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
import org.openimaj.feature.DoubleFV;
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
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;


public class Run3Test {

    //TODO check/play around with these numbers and all numbers that aren't variables
    final static int MAX_FEATURES = 10000;
    final static int CLUSTERS = 500;

    public static void main(String[] args) throws IOException {
        Path trainingDataPath = Paths.get("resources/training/");
        Path testingDataPath = Paths.get("resources/testing/");
        VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<>(trainingDataPath.toAbsolutePath().toString(), ImageUtilities.FIMAGE_READER);
        VFSListDataset<FImage> testingData = new VFSListDataset<>(testingDataPath.toAbsolutePath().toString(), ImageUtilities.FIMAGE_READER);

        //Split training data into training and testing subsets
        GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<>(trainingData, 50, 0, 50);
        GroupedDataset<String, ListDataset<FImage>, FImage> trainingSplit = splits.getTrainingDataset();
        GroupedDataset<String, ListDataset<FImage>, FImage> testingSplit = splits.getTestDataset();

        DenseSIFT dsift = new DenseSIFT(5, 7);
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<>(dsift, 6f, 7);

        HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(trainingSplit, 100), pdsift);

        //Construct and train linear classifier that uses a homogeneous kernel map
        PHOWExtractor extractor = new PHOWExtractor(pdsift, assigner);
        HomogeneousKernelMap homogeneousKernelMap = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
        LiblinearAnnotator<FImage, String> classifier = new LiblinearAnnotator<>(homogeneousKernelMap.createWrappedExtractor(extractor), LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        classifier.train(trainingSplit);
        System.out.println("classifier trained");

        //TODO not meaningful yet, need to measure accuracy on testingSplit and then produce run3.txt
        ClassificationEvaluator<CMResult<String>, String, FImage> eval = new ClassificationEvaluator<>(classifier, testingSplit, new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));
        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);
        System.out.println("done");
    }


     static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>, FImage> trainingSplit, PyramidDenseSIFT<FImage> pdsift) {
        List<LocalFeatureList<ByteDSIFTKeypoint>> features = new ArrayList<>();

        for (String directory : trainingSplit.getGroups()) {
            ListDataset<FImage> directoryImages = trainingSplit.getInstances(directory);
            for (FImage image : directoryImages) {
                pdsift.analyseImage(image);
                features.add(pdsift.getByteKeypoints(0.005f));
            }
        }

        //TODO - probably not needed?
        if (features.size() > MAX_FEATURES) {
            features = features.subList(0, MAX_FEATURES);
        }

        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(CLUSTERS);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<>(features);
        ByteCentroidsResult result = km.cluster(datasource);

         System.out.println("returning assigner");
        return result.defaultHardAssigner();
    }

    static class PHOWExtractor implements FeatureExtractor<SparseIntFV, FImage> {
        PyramidDenseSIFT<FImage> pdsift;
        HardAssigner<byte[], float[], IntFloatPair> assigner;

        public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner) {
            this.pdsift = pdsift;
            this.assigner = assigner;
        }

        //TODO try using BlockSpatialAggregator or PyramidSpatialAggregator and compare accuracy?
        public SparseIntFV extractFeature(FImage image) {
            pdsift.analyseImage(image);
            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);
            System.out.println("returning bovw");
            return bovw.aggregate(pdsift.getByteKeypoints(0.015f));
        }
    }




}
