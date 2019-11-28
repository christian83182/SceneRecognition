package uk.ac.soton.ecs;

import java.io.*;
import java.nio.file.Path;

public class Utils {
    /**
     * Computes an accuracy rating using a correct and prediction file.
     * @param correctData The Path to the file with the correct answers.
     * @param predictedData The Path to the file with the incorrect answers.
     * @return A number between 0 and 1 depicting the accuracy.
     * @throws IOException If the files cannot be read.
     */
    public static double computeAccuracy(Path correctData, Path predictedData) throws IOException {
        File correctFile = correctData.toFile();
        File predicatedFile = predictedData.toFile();

        BufferedReader correctReader = new BufferedReader(new FileReader(correctFile));
        BufferedReader predictedReader = new BufferedReader(new FileReader(predicatedFile));

        String correctLine;
        String predictedLine;
        Double correct = 0.0;
        Double total = 0.0;

        while((correctLine = correctReader.readLine()) != null && (predictedLine = predictedReader.readLine()) != null){
            if( correctLine.equals(predictedLine)){
                correct++;
            }
            total++;
        }

        return correct/total;
    }
}
