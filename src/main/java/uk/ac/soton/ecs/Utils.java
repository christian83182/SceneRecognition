package uk.ac.soton.ecs;

import java.io.*;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

public class Utils {
    /**
     * Computes an accuracy rating using a correct and prediction file.
     * @param correctData The Path to the file with the correct answers.
     * @param predictedData The Path to the file with the incorrect answers.
     * @return A number between 0 and 1 depicting the accuracy.
     * @throws IOException If the files cannot be read.
     */
    public static double computeAccuracy(Path correctData, Path predictedData) throws IOException {
        return computeAccuracy(correctData, createMapFromFile(predictedData));
    }

    public static double computeAccuracy(Path correctData, Map<String,String> classificationMap) throws IOException {
        return computeAccuracy(createMapFromFile(correctData), classificationMap);
    }

    public static double computeAccuracy(Map<String, String> correctClassMap, Map<String,String> classificationMap){
        Double correct = 0.0;
        Double total = 0.0;

        for(String key : classificationMap.keySet()){
            if(correctClassMap.containsKey(key)){
                if(correctClassMap.get(key).equals(classificationMap.get(key))){
                    correct++;
                }
                total++;
            }
        }
        return correct/total;
    }

    private static Map<String,String> createMapFromFile(Path pathToFile) throws IOException {
        Map<String,String> classMap = new HashMap<>();
        File file = pathToFile.toFile();
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        while((line = reader.readLine()) != null){
            String[] values = line.split(" ");
            classMap.put(values[0],values[1]);
        }
        return classMap;
    }
}