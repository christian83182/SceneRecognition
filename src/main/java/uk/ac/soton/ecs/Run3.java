package uk.ac.soton.ecs;

import ch.akuhn.matrix.Vector;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInitDistribution;

import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;
import java.util.stream.Stream;

public class Run3 {
    private final static Logger logger = LoggerFactory.getLogger(Run3.class);
    private final static Double SIMILARITY_THRESHOLD = 0.4;
    private static int height = 256;
    private static int width = 256;
    private static int channels = 1;
    private static int batchSize = 16;
    private static int numOfLabels = 15;
    private static int epochs = 10;

    private static final int randomSeed = 2019;
    private static Random random = new Random(randomSeed);

    public static void main(String[] args) throws IOException {
        Run3 run3 = new Run3();
        run3.run();
    }

    private void run() throws IOException {
        File trainingData = new File("resources/training/");
        File testingData = new File("resources/testingSubset/");

        //Define some FileSplits which will randomize the order of data as it's fed into the CNN.
        FileSplit trainingFileSplit = new FileSplit(trainingData, NativeImageLoader.ALLOWED_FORMATS, random);

        //A class which labels data according to its folder structure.
        ParentPathLabelGenerator trainingLabelMaker = new ParentPathLabelGenerator();

        //A class which will read images from the given FileSplits.
        ImageRecordReader trainingRecordReader = new ImageRecordReader(height, width, channels, trainingLabelMaker);
        trainingRecordReader.initialize(trainingFileSplit);

        //Create an iterator to iterate over the images in the training and testing dataset.
        DataSetIterator trainingIterator = new RecordReaderDataSetIterator(trainingRecordReader, batchSize, 1, numOfLabels);

        //Create and add an image preprocessor which normalizes the value of the pixels to values between 0 and 1.
        DataNormalization normalizer = new ImagePreProcessingScaler(0,1);
        normalizer.fit(trainingIterator);
        trainingIterator.setPreProcessor(normalizer);

        logger.info("--- BUILDING MODEL ---");
        double nonZeroBias = 1;
        double dropOut = 0.5;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(randomSeed)
                .weightInit(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .updater(new AdaDelta())
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .l2(5 * 1e-4)
                .list()
                .layer(convInit("cnn1", channels, 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
                .layer(new LocalResponseNormalization.Builder().name("lrn1").build())
                .layer(maxPool("maxpool1", new int[]{3,3}))
                .layer(conv5x5("cnn2", 256, new int[] {1,1}, new int[] {2,2}, nonZeroBias))
                .layer(new LocalResponseNormalization.Builder().name("lrn2").build())
                .layer(maxPool("maxpool2", new int[]{3,3}))
                .layer(conv3x3("cnn3", 384, 0))
                .layer(conv3x3("cnn4", 384, nonZeroBias))
                .layer(conv3x3("cnn5", 256, nonZeroBias))
                .layer(maxPool("maxpool3", new int[]{3,3}))
                .layer(fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new NormalDistribution(0, 0.005)))
                .layer(fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new NormalDistribution(0, 0.005)))
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numOfLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));

        //train the model by calling fit() epoch times.
        for (int i = 0; i < epochs; i++) {
            model.fit(trainingIterator);
            System.out.println("Epoch " + i +" Complete");
        }

        //Following similar steps to creating a DataSetIterator for the training data.
        //In this case, a custom FileLabelGenerator class is used which labels testing data according to correct.txt
        FileSplit testingFileSplit = new FileSplit(testingData, NativeImageLoader.ALLOWED_FORMATS, random);
        FileLabelGenerator testingLabelMaker = new FileLabelGenerator();
        ImageRecordReader testingRecordReader = new ImageRecordReader(height, width, channels, testingLabelMaker);
        testingRecordReader.initialize(testingFileSplit);
        DataSetIterator testingIterator = new RecordReaderDataSetIterator(testingRecordReader, batchSize, 1, numOfLabels);
        normalizer.fit(testingIterator);
        testingIterator.setPreProcessor(normalizer);

        Evaluation evaluation = model.evaluate(testingIterator);
        System.out.println(evaluation.stats());
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
        return firstVectorNormal.compare(secondVectorNormal, DoubleFVComparison.EUCLIDEAN);
    }

    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    private SubsamplingLayer maxPool(String name,  int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }

    private DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).weightInit(new WeightInitDistribution(dist)).build();
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
            return true;
        }
    }
}
