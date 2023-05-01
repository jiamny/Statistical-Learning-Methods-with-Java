package com.github.jiamny.ml.Ch03_Data_retrieving_and_storing;

import java.lang.String;
import smile.io.Read;
import smile.plot.swing.ScatterPlot;

public class ArffFileDataSmile {
    public static void main(String [] args) {
        try {
            var iris = Read.arff("data/iris.arff");
            var canvas = ScatterPlot.of(iris, "sepallength", "sepalwidth", "class", '*').canvas();
            canvas.setAxisLabels("sepallength", "sepalwidth");
            canvas.setTitle("Iris data");
            canvas.window();
            Thread.sleep(2000);
            canvas.clear();
        } catch(Exception e) {
            e.printStackTrace();
        }
       System.exit(0);
    }
}
