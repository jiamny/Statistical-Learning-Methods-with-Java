package com.github.jiamny.ml.utils;

import org.apache.commons.csv.CSVFormat;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import smile.data.DataFrame;
import smile.io.CSV;
import smile.io.Read;
import smile.math.matrix.Matrix;

import java.nio.file.Paths;
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

    public static int [] range(int start, int end) {
        int size = (end - start);
        int [] rg = new int[size];
        for( int i = start; i < end; i++ )
            rg[i-start] = i;
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

    public static int bisectLeft(double[] nums, double target) {
        int i = 0;
        int j = nums.length - 1;
        while (i <= j) {
            int m = i + (j-i) / 2;
            if (nums[m] >= target) {
                j = m - 1;
            } else {
                i = m + 1;
            }
        }
        return i;
    }

    public static int bisectRight(double [] nums, double target) {
        int i = 0;
        int j = nums.length - 1;
        while (i <= j) {
            int m = i + (j-i) / 2;
            if (nums[m] <= target) {
                i = m + 1;
            } else {
                j = m - 1;
            }
        }
        return j+1;
    }

    public static double roundAvoid(double value, int places) {
        double scale = Math.pow(10, places);
        return Math.round(value * scale) / scale;
    }

    public static int [] doubleToInt(double [] data) {
        int [] res = new int[data.length];
        for( int i = 0; i < data.length; i++ )
            res[i] = (int) data[i];
        return res;
    }

    public static DataFrame loadIrisData(String fileName) {
        // ----------------------------------------------
        DataFrame iris_data = null;
        try {
            CSV rd = new CSV();
            iris_data = rd.read(Paths.get(fileName));
            System.out.println(iris_data.schema());
            String [] names = {"sepallength","sepalwidth","petallength","petalwidth"};
            String [] cnames = {"class"};

            int [] label_idx = new int[1];
            label_idx[0] = iris_data.ncol() - 1;
            DataFrame irisClass = iris_data.select(label_idx);

            int [][] c_data = new int[iris_data.nrow()][1];
            String c = "";
            int cls = -1;
            for(int i = 0; i < iris_data.nrow(); i++ ) {
                if( irisClass.getString(i, 0).equalsIgnoreCase(c) )
                    c_data[i][0] = cls;
                else {
                    cls++;
                    c = irisClass.getString(i, 0);
                    c_data[i][0] = cls;
                }
            }
            irisClass = DataFrame.of(c_data, cnames);

            int [] data_idx = new int[iris_data.ncol()-1];
            for(int i = 0; i < (iris_data.ncol()-1); i++ )
                data_idx[i] = i;
            DataFrame data = iris_data.select(data_idx);

            iris_data = DataFrame.of(data.toArray(), names);
            iris_data = iris_data.merge(irisClass);
        } catch(Exception e) {
            e.printStackTrace();
        }
        return iris_data;
    }

    public static double euler_distance(Matrix point1, Matrix point2) {
        // 计算两点之间的欧拉距离，支持多维
        Matrix d = point1.sub(point2);
        double distance = 0.0;
        if( d.nrow() > 1 ) {
            // tranpose  1xn => nx1
            distance = d.mm(d.transpose()).get(0, 0);
        } else {
            // tranpose  nx1 => 1xn
            distance = d.transpose().mm(d).get(0, 0);
        }
        //for a, b in zip(point1, point2):
        //distance += math.pow(a - b, 2)
        return Math.sqrt(distance);
    }
}
