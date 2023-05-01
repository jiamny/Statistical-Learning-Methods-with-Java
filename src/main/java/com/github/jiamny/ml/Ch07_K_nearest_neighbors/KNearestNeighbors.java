package com.github.jiamny.ml.Ch07_K_nearest_neighbors;

import com.github.jiamny.ml.utils.StatisticHelper;
import smile.data.vector.IntVector;
import smile.io.Read;
import smile.plot.swing.ScatterPlot;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static com.github.jiamny.ml.utils.DataFrameHelper.unique;


public class KNearestNeighbors {
    // ----------------------------
    // distance
    // p=1 曼哈顿距离
    // p=2 欧氏距离
    // p=∞ 切比雪夫距离
    // ----------------------------
    public double L(double[] x, double[] y, double p){
        // x1 = [1, 1], x2 = [5,1]
        if( (x.length == y.length) && (x.length > 1) ){
            double sum = 0.;
            for( int i = 0; i < x.length; i++ ) {
                sum += Math.pow(Math.abs(x[i] - y[i]), p);
            }
            return Math.pow(sum, 1.0 / p);
        } else {
            return 0.;
        }
    }

    public static void main(String [] args) {
        KNearestNeighbors knn = new KNearestNeighbors();
        // test
        double[] x1 = {1., 1.};
        Double[] x2 = {5., 1.};
        Double[] x3 = {4., 4.};
        ArrayList<ArrayList<Double>> xs = new ArrayList<>();
        ArrayList<Double> xx2 = new ArrayList<>(Arrays.asList(x2));
        xs.add(xx2);
        ArrayList<Double> xx3 = new ArrayList<>(Arrays.asList(x3));
        xs.add(xx3);

        // x1, x2
        for(int i = 1; i < 5; i++ ) {
            //System.out.println("----------------------------------------> " + i);
            HashMap<String, Double> r = new HashMap<>();
            for( int c = 0; c < xs.size(); c++ ) {
                String dsc = "1 - " + xs.get(c).toString();
                double val = knn.L(x1, xs.get(c).stream().mapToDouble(j->j).toArray(), i*1.0);
                //System.out.println(dsc + " => " + val);
                r.put(dsc, val);
            }
            //var nval = StatisticHelper.min(r, Double::compare);
            Map.Entry<String, Double> nval = StatisticHelper.maxOrmin(r, Double::compare, false);
            System.out.println(nval.getValue() + " " + nval.getKey());
        }
        // 遍历所有数据点，找出n个距离最近的点的分类情况，少数服从多数

        //data
        try {
            var iris_df = Read.arff("data/iris.arff");
            System.out.println(iris_df.column("class"));
            System.out.println(iris_df.summary());

            String[] cls = iris_df.column("class").toStringArray();
            String [] udt = Arrays.stream(unique(cls))
                    .toArray(String[]::new);
            System.out.println(Arrays.deepToString(udt));

            //var newdf = iris_df.drop("class");
            int [] cl = new int[cls.length];

            for( int i = 0; i < iris_df.nrow(); i++ ) {
                cl[i] = 0;
                if( cls[i].equalsIgnoreCase("Iris-setosa") )  cl[i] = 1;
                if( cls[i].equalsIgnoreCase("Iris-versicolor") ) cl[i] = 2;
                if( cls[i].equalsIgnoreCase("Iris-virginica") ) cl[i] = 3;
            }
            var mdf = iris_df.merge(IntVector.of("label", cl));

            var canvas = ScatterPlot.of(mdf, "sepallength", "sepalwidth", "class", '*').canvas();
            canvas.setAxisLabels("sepallength", "sepalwidth");
            canvas.window();

        }catch(Exception e) {
            e.printStackTrace();
        }

    }
}
