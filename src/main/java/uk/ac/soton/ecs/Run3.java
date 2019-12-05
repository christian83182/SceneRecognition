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
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
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
    private static int batchSize = 128;
    private static int outputNum = 15;
    private static int epochs = 50;

    private static final int randomSeed = 2019;
    private static Random random = new Random(randomSeed);

    public static void main(String[] args) throws IOException {
        Run3 run3 = new Run3();
        run3.run();
    }

    private void run() throws IOException {
        File trainingData = new File("resources/training/");
        File testingData = new File("resources/testing/");

        //Define some FileSplits which will randomize the order of data as it's fed into the CNN.
        FileSplit trainingFileSplit = new FileSplit(trainingData, NativeImageLoader.ALLOWED_FORMATS, random);

        //A class which labels data according to its folder structure.
        ParentPathLabelGenerator trainingLabelMaker = new ParentPathLabelGenerator();

        //A class which will read images from the given FileSplits.
        ImageRecordReader trainingRecordReader = new ImageRecordReader(height, width, channels, trainingLabelMaker);
        trainingRecordReader.initialize(trainingFileSplit);

        //Create an iterator to iterate over the images in the training and testing dataset.
        DataSetIterator trainingIterator = new RecordReaderDataSetIterator(trainingRecordReader, batchSize, 1, outputNum);

        //Create and add an image preprocessor which normalizes the value of the pixels to values between 0 and 1.
        DataNormalization normalizer = new ImagePreProcessingScaler(0,1);
        normalizer.fit(trainingIterator);
        trainingIterator.setPreProcessor(normalizer);

        logger.info("--- BUILDING MODEL ---");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(randomSeed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.006, 0.9))
                .l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(height * width)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(100)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);

        model.setListeners(new ScoreIterationListener(10));

        logger.info("--- TRAINING MODEL ---");
        for (int i = 0; i < epochs; i++) {
            model.fit(trainingIterator);
            System.out.println("Epoch " + i +" Complete");
        }

        logger.info("--- EVALUATING MODEL (WIP) ---");

        //Following similar steps to creating a DataSetIterator for the training data.
        //In this case, a custom FileLabelGenerator class is used which labels testing data according to correct.txt
        FileSplit testingFileSplit = new FileSplit(testingData, NativeImageLoader.ALLOWED_FORMATS, random);
        FileLabelGenerator testingLabelMaker = new FileLabelGenerator();
        ImageRecordReader testingRecordReader = new ImageRecordReader(height, width, channels, testingLabelMaker);
        testingRecordReader.initialize(testingFileSplit);
        DataSetIterator testingIterator = new RecordReaderDataSetIterator(testingRecordReader, batchSize, 1, outputNum);
        normalizer.fit(testingIterator);
        testingIterator.setPreProcessor(normalizer);
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
