package com.github.jiamny.ml.Ch03_Data_retrieving_and_storing;

import org.apache.commons.csv.CSVFormat;
import smile.io.Read;

public class TextAndCsvDataSmile {

    public static void main(String [] args) {
        // ----------------------------------------------
        try {
            var format = CSVFormat.newFormat(',');
            var mnist_train = Read.csv("./data/data_cleaned.csv", format);
            System.out.println(mnist_train.get(3));
        } catch(Exception e) {
            e.printStackTrace();
        }
    }
}
