package com.github.jiamny.ml.Ch03_Data_retrieving_and_storing;

import smile.data.DataFrame;
import smile.io.CSV;

public class TextFileEdaSmile {

    public static void main(String [] args) {

        try {
            CSV csv = new CSV();
            DataFrame dt = csv.read("./data/data_cleaned.csv");
            System.out.println(dt.summary());

        } catch(Exception e) {
            e.printStackTrace();
        }
    }
}
