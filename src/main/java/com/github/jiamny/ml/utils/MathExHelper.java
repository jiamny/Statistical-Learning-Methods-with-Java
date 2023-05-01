package com.github.jiamny.ml.utils;

public class MathExHelper {

    public static double[][] arrayBindByCol(double[] ... list) {

        int n = 0;
        for (double[] x : list) {
            n = x.length;
            break;
        }
        double[][] y = new double[n][list.length];
        int pos = 0;
        for (double[] x: list) {
            //System.arraycopy(x, 0, y, 0, x.length);
            for(int i = 0; i < n; i++)
                y[i][pos] = x[i];
            pos += 1; // x.length;
        }
        return y;
    }

 //   public static double[] poly1d(double[] x, double ... list) {
 //       int max_degree = x.length - 1;
 //   }
}
