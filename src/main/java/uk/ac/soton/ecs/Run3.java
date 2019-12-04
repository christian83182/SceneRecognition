package uk.ac.soton.ecs;

import ch.akuhn.matrix.Vector;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;

import java.nio.file.Path;
import java.nio.file.Paths;

public class Run3 {

    //A Threshold used in determining if similar images should be considered the same image.
    public static Double SIMILARITY_THRESHOLD = 0.4;

    public static void main(String[] args) throws FileSystemException {
        Path trainingDataPath = Paths.get("resources/training/");
        Path testingDataPath = Paths.get("resources/testing/");
        VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<>(trainingDataPath.toAbsolutePath().toString(), ImageUtilities.FIMAGE_READER);
        VFSListDataset<FImage> testingData = new VFSListDataset<>(testingDataPath.toAbsolutePath().toString(), ImageUtilities.FIMAGE_READER);

        //do something
    }

    private static boolean isSameImage(FImage firstImage, FImage secondImage){
        //Determines if two images are similar by comparing their similarity score against the threshold.
        return getSimilarityScore(firstImage, secondImage) < SIMILARITY_THRESHOLD;
    }

    private static double getSimilarityScore(FImage firstImage, FImage secondImage){
        //Resize both images to the same size so their pixel vectors can be compared.
        ResizeProcessor resizeProcessor = new ResizeProcessor(256,256,false);
        resizeProcessor.processImage(firstImage);
        resizeProcessor.processImage(secondImage);

        //Create two Vector object from the image's pixel values.
        Vector firstVector = Vector.wrap(firstImage.getDoublePixelVector());
        Vector secondVector = Vector.wrap(secondImage.getDoublePixelVector());

        //Center and normalize both vectors.
        firstVector.applyCentering();
        firstVector = firstVector.times(1/firstVector.norm());
        secondVector.applyCentering();
        secondVector = secondVector.times(1/secondVector.norm());

        //Convert both to feature vectors to use their compare() method can be used.
        DoubleFV firstVectorNormal = new DoubleFV(firstVector.unwrap());
        DoubleFV secondVectorNormal = new DoubleFV(secondVector.unwrap());

        //Return the euclidean distance between both vectors using the DoubleFX.compare() function.
        return firstVectorNormal.compare(secondVectorNormal,DoubleFVComparison.EUCLIDEAN);
    }
}
