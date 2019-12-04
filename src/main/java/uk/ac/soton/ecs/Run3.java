package uk.ac.soton.ecs;

import ch.akuhn.matrix.Vector;
import org.apache.commons.vfs2.FileSystemException;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

public class Run3 {
    private final static Logger logger = LoggerFactory.getLogger(Run3.class);
    private final static Double SIMILARITY_THRESHOLD = 0.4;
    private static int height = 256;
    private static int width = 256;
    private static int channels = 1;
    private static int batchSize = 15;
    private static int outputNum = 15;
    private static int epochs = 50;
    private static Random random = new Random(2019);

    public static void main(String[] args) throws IOException {
        Run3 run3 = new Run3();
        run3.run();
    }

    private void run() throws IOException {
        File trainingData = new File("resources/training/");
        File testingData = new File("resources/testing/");

        //Define some FileSplits which will randomize the order of data as it's fed into the CNN.
        FileSplit trainingFileSplit = new FileSplit(trainingData, NativeImageLoader.ALLOWED_FORMATS, random);
        FileSplit testingFileSplit = new FileSplit(testingData, NativeImageLoader.ALLOWED_FORMATS, random);

        //A class which labels data according to its folder structure.
        ParentPathLabelGenerator trainingLabelMaker = new ParentPathLabelGenerator();

        //A custom class which labels testing data according to the labels in correct.txt.
        FileLabelGenerator testingLabelMaker = new FileLabelGenerator();

        //A class which will read images from the given FileSplits.
        ImageRecordReader trainingRecordReader = new ImageRecordReader(height, width, channels, trainingLabelMaker);
        ImageRecordReader testingRecordReader = new ImageRecordReader(height, width, channels, testingLabelMaker);
        trainingRecordReader.initialize(trainingFileSplit);
        testingRecordReader.initialize(testingFileSplit);

        //Create an iterator to iterate over the images in the training dataset.
        DataSetIterator trainingIterator = new RecordReaderDataSetIterator(trainingRecordReader, batchSize, 1, outputNum);

        //Run one batch and output the results in order to verify the labeler is working correctly.
        trainingRecordReader.setListeners(new LogRecordListener());
        for(int i =0; i < 1; i++){
            DataSet dataSet = trainingIterator.next();
            System.out.println(dataSet);
            System.out.println(trainingIterator.getLabels());
        }
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

    //A custom class which will label data according to the correct.txt file in the resources folder.
    static class FileLabelGenerator implements PathLabelGenerator{
        @Override
        public Writable getLabelForPath(String pathAsString) {
            return  getLabelFromResource(new File(pathAsString));
        }

        @Override
        public Writable getLabelForPath(URI uri) {
            return getLabelFromResource(new File(uri));
        }

        //A method to label a given testing file. Using the filename as an index, lookup its label in correct.txt
        private Writable getLabelFromResource(File file){
            Integer lineIndex = Integer.parseInt(file.getName().replace(".jpg",""));
            String fileLine = "";

            try (Stream<String> all_lines = Files.lines(Paths.get("resources/results/correct.txt"))) {
                fileLine = all_lines.skip(lineIndex).findFirst().get();
            } catch (IOException e) {
                System.err.print("Attempting to create label for unlabelled data: " + file.getName());
            }

            String label = fileLine.split(" ")[1];
            return new Text(label);
        }

        @Override
        public boolean inferLabelClasses() {
            return false;
        }
    }
}
