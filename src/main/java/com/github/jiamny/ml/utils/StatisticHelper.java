package com.github.jiamny.ml.utils;

import java.util.*;
import java.util.function.Function;
import smile.math.MathEx;

public class StatisticHelper {
    public static <T extends Comparable<T>> T mode(T[] a, T initV) {
        T maxValue = initV;
        int maxCount = 0, i, j;
        int n = a.length;

        for (i = 0; i < n; ++i) {
            int count = 0;
            for (j = 0; j < n; ++j) {
                if (a[j].compareTo(a[i]) == 0 )
                    ++count;
            }

            if (count > maxCount) {
                maxCount = count;
                maxValue = a[i];
            }
        }
        return maxValue;
    }

    public static void printVectorElements(double [] ax) {
        Arrays.stream(ax).forEach(num -> System.out.print(num + " "));
        System.out.println();
    }

    public static void printVectorElements(int [] ax) {
        Arrays.stream(ax).forEach(num -> System.out.print(num + " "));
        System.out.println();
    }

    public static void printVectorObjects(Object [] ax) {
        Arrays.stream(ax).forEach(num -> System.out.print(num + " "));
        System.out.println();
    }

    public static ArrayList<Double> cumulativeSum(ArrayList<Double> numbers) {
        // variable
        double sum = 0.0;

        // traverse through the array
        for (int i = 0; i < numbers.size(); i++) {
            sum += numbers.get(i); // find sum
            numbers.set(i, sum);   // replace
        }
        // return
        return numbers;
    }

    public static <K, V> Map.Entry<K, V> maxOrmin(Map<K, V> map, Comparator<V> comp, boolean max) {
        Iterator<Map.Entry<K, V>> entries = map.entrySet().iterator();

        if (!entries.hasNext()) {
            return null;
        }
        Map.Entry<K, V> mv;
        for (mv = entries.next(); entries.hasNext();) {
            Map.Entry<K, V> value = entries.next();
            if( max ) {
                if (comp.compare(value.getValue(), mv.getValue()) > 0) {
                    mv = value;
                }
            } else {
                if (comp.compare(value.getValue(), mv.getValue()) < 0) {
                    mv = value;
                }
            }

        }
        return mv;
    }

    public static double[] linespace(double min, double max, int points) {
        double[] d = new double[points];
        for (int i = 0; i < points; i++){
            d[i] = min + i * (max - min) / (points - 1);
        }
        return d;
    }

    public static double[][]  impute(Function<double[][], double[][]> imputer, double[][] data) throws Exception {
        MathEx.setSeed(19650218); // to get repeatable results.
/*
        int n = 0;
        double[][] missing = new double[data.length][data[0].length];
        for (int i = 0; i < missing.length; i++) {
            for (int j = 0; j < missing[i].length; j++) {
                if (MathEx.random() < rate) {
                    n++;
                    missing[i][j] = Double.NaN;
                } else {
                    missing[i][j] = data[i][j];
                }
            }
        }
*/
        double[][] imputed = imputer.apply(data);

        double error = 0.0;
        for (int i = 0; i < imputed.length; i++) {
            for (int j = 0; j < imputed[i].length; j++) {
                error += Math.abs(data[i][j] - imputed[i][j]) / data[i][j];
            }
        }
        System.out.println("error: " + error);
        return imputed;
    }

    public static int maxIndex(double [] data) {
        int idx = -1;
        double max = Double.NEGATIVE_INFINITY;
        for( int i = 0; i < data.length; i++ ) {
            if( max < data[i] ) {
                idx = i;
                max = data[i];
            }
        }
        return idx;
    }
}
