package uk.ac.soton.ecs;

import ch.akuhn.matrix.Vector;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.AlexNet;
import org.deeplearning4j.zoo.model.ResNet50;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.model.VGG19;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
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
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

public class Run3 {
    private final static Logger logger = LoggerFactory.getLogger(Run3.class);
    private final static Double SIMILARITY_THRESHOLD = 0.4;
    private static int height = 224;
    private static int width = 224;
    private static int channels = 3;
    private static int batchSize = 24;
    private static int numOfLabels = 15;
    private static int epochs = 1000;

    private static final int randomSeed = 2019;
    private static Random random = new Random(randomSeed);

    public static void main(String[] args) throws IOException {
        System.setProperty("org.deeplearning4j.resources.baseurl","https://dl4jdata.blob.core.windows.net/");

        Run3 run3 = new Run3();
        run3.run();
    }

    private void run() throws IOException {
        File trainingData = new File("resources/training/");
        File testingData = new File("resources/testingSubset/");

        //Define some FileSplits which will randomize the order of data as it's fed into the CNN.
        FileSplit testingFileSplit = new FileSplit(testingData, NativeImageLoader.ALLOWED_FORMATS, random);
        //A class which labels data according to its label in correct.txt
        FileLabelGenerator testingLabelMaker = new FileLabelGenerator();
        //A class which will read images from the given FileSplits.
        ImageRecordReader testingRecordReader = new ImageRecordReader(height, width, channels, testingLabelMaker);
        testingRecordReader.initialize(testingFileSplit);
        //Create an iterator to iterate over the images in the training and testing dataset.
        DataSetIterator testingIterator = new RecordReaderDataSetIterator(testingRecordReader, batchSize, 1, numOfLabels);
        //Create and add an image preprocessor which normalizes the value of the pixels to values between 0 and 1.
        DataNormalization normalizer = new ImagePreProcessingScaler(0,1);
        normalizer.fit(testingIterator);
        testingIterator.setPreProcessor(normalizer);

        List<Pair<ImageTransform,Double>> transformList = Arrays.asList(
                new Pair<>(new FlipImageTransform(1),0.5),
                new Pair<>(new EqualizeHistTransform(),0.5),
                new Pair<>(new CropImageTransform(random, 50), 0.5));
        PipelineImageTransform pipeline = new PipelineImageTransform(transformList,true);

        FileSplit trainingFileSplit = new FileSplit(trainingData, NativeImageLoader.ALLOWED_FORMATS, random);
        ParentPathLabelGenerator trainingLabelMaker = new ParentPathLabelGenerator();
        ImageRecordReader trainingRecordReader = new ImageRecordReader(height, width, channels, trainingLabelMaker);
        trainingRecordReader.initialize(trainingFileSplit,pipeline);
        DataSetIterator trainingIterator = new RecordReaderDataSetIterator(trainingRecordReader, batchSize, 1, numOfLabels);
        normalizer.fit(trainingIterator);
        trainingIterator.setPreProcessor(normalizer);

        logger.info("--- BUILDING MODEL ---");
        ComputationGraph model = getModel();
        model.init();

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));

        //train the model by calling fit() epoch times.
        for (int i = 0; i < epochs; i++) {
            model.fit(trainingIterator);
            logger.info("Completed Epoch: " + (i));
            if(i %5 == 0){
                Evaluation eval = model.evaluate(testingIterator);
                logger.info(eval.stats(false, true));
                testingIterator.reset();
            }
            trainingIterator.reset();
        }
    }

    private ComputationGraph getModel() throws IOException {
        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
        logger.info(vgg16.summary());

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .updater(new Nesterovs(5e-5))
                .seed(randomSeed)
                .build();

        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("fc2")
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096).nOut(numOfLabels)
                                .weightInit(new NormalDistribution(0,0.2*(2.0/(4096+numOfLabels))))
                                .activation(Activation.SOFTMAX).build(),
                        "fc2")
                .build();
        logger.info(vgg16Transfer.summary());

        return vgg16Transfer;
    }

    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    private SubsamplingLayer maxPool(String name,  int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
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
