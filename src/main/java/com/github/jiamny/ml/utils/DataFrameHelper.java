package com.github.jiamny.ml.utils;

import org.apache.commons.csv.CSVFormat;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import smile.data.DataFrame;
import smile.io.Read;
import smile.math.matrix.Matrix;

import java.util.*;
import java.util.stream.DoubleStream;

public class DataFrameHelper {

    public static boolean hasDataMissing(Matrix data) {

        boolean hasMissing = false;
        for( int r = 0; r < data.nrow(); r++ ) {
            for( int c = 0; c < data.nrow(); c++ ) {
                if( Double.isNaN(data.get(r,c)) || Double.isInfinite(data.get(r, c)) ) {
                    hasMissing = true;
                    break;
                }
            }
        }
        return hasMissing;
    }

    public static int [] range(int size) {
        int [] rg = new int[size];
        for( int i = 0; i < size; i++ )
            rg[i] = i;
        return rg;
    }


    /**
     * Convert double array to int array
     * @param array2
     * @return
     */
    public static int[] roundUp2(double[] array2) {
        return DoubleStream.of(array2).mapToInt(d -> (int) Math.ceil(d)).toArray();
    }

    public static Object[] unique(Object [] v) {
        Map<Object, Integer> udt = new HashMap<>();

        for(Object obj : v) {
            if( ! udt.containsKey(obj) )
                udt.put(obj, 1);
        }
        return udt.keySet().toArray(new Object[0]);
    }

    public static void loadMnistData(String fileName, ArrayList<DataFrame> tdt) {
        try {
            var format = CSVFormat.newFormat(',');
            DataFrame mnist_train = Read.csv(fileName, format);

            int [] label_idx = new int[1];
            label_idx[0] = 0;
            DataFrame train_labels = mnist_train.select(label_idx);

            int [] data_idx = new int[mnist_train.ncol()-1];
            for(int i = 1; i <= (mnist_train.ncol()-1); i++ )
                data_idx[i - 1] = i;
            DataFrame train_data = mnist_train.select(data_idx);

            //在放入的同时将原先字符串形式的数据转换为整型
            //此外将数据进行了二值化处理，大于128的转换成1，小于的转换成0，方便后续计算
            //double [][] dt = train_data.toMatrix().toArray();
            int [][] mdt = new int[train_data.nrow()][train_data.ncol()];

            for(int r = 0; r < train_data.nrow(); r++) {
                for(int c = 0; c < train_data.ncol(); c++) {
                    if(train_data.getDouble(r, c) > 128) {
                        //System.out.println("r: " + r + " c: " + c + " V: " + train_data.getDouble(r, c));
                        mdt[r][c] = 1;
                    } else {
                        mdt[r][c] = 0;
                    }
                }
            }
            DataFrame ntrain_data = DataFrame.of(mdt);

            //System.out.println("r: " + 59999 + " c: " + 595 + " V: " + ntrain_data.get(59999, 595));
            // return processed data
            tdt.add(ntrain_data);
            tdt.add(train_labels);
        } catch(Exception e) {
            e.printStackTrace();
        }
    }
}
