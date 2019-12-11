package uk.ac.soton.ecs;

import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.kernel.HomogeneousKernelMap;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;


public class Run3Test {

    public static void main(String[] args) throws IOException {
        Path trainingDataPath = Paths.get("resources/training/");
        Path testingDataPath = Paths.get("resources/testing/");
        VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<>(trainingDataPath.toAbsolutePath().toString(), ImageUtilities.FIMAGE_READER);
        VFSListDataset<FImage> testingData = new VFSListDataset<>(testingDataPath.toAbsolutePath().toString(), ImageUtilities.FIMAGE_READER);

        //Training with a linear classifier that uses a homogeneous kernel map
        DenseSIFT dsift = new DenseSIFT();
        DenseSIFTExtractor extractor = new DenseSIFTExtractor(dsift);
        HomogeneousKernelMap homogeneousKernelMap = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
        LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<>(homogeneousKernelMap.createWrappedExtractor(extractor), LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        ann.train(trainingData);
    }

    //TODO Is the extracted feature DoubleFV?
    static class DenseSIFTExtractor implements FeatureExtractor<DoubleFV, FImage> {
        DenseSIFT dsift;

        public DenseSIFTExtractor(DenseSIFT dsift) {
            this.dsift = dsift;
        }

        @Override
        public DoubleFV extractFeature(FImage image) {
            dsift.analyseImage(image);
            return null;
        }
    }




}
