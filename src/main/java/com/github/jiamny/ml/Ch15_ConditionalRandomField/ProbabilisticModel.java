package com.github.jiamny.ml.Ch15_ConditionalRandomField;

import smile.data.DataFrame;
import smile.math.Random;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static com.github.jiamny.ml.utils.DataFrameHelper.*;

public class ProbabilisticModel {
    // 转移特征函数
    public  int t1(int y0, int y1, int[] x, int i) {
        Set<Integer> Y0 = new HashSet<>();
        Set<Integer> Y1 = new HashSet<>();
        Set<Integer> IS = new HashSet<>();
        Y0.add(0);
        Y1.add(1);
        IS.add(1);
        IS.add(2);
        int[][] xsdt = {{0, 1, 0}, {0, 1, 1}, {0, 0, 0}, {0, 0, 1}};
        Set<String> XS = new HashSet<>();
        for (int j = 0; j < xsdt.length; j++)
            XS.add(Arrays.toString(xsdt[j]));
        if (Y0.contains(y0) && Y1.contains(y1) && XS.contains(Arrays.toString(x)) && IS.contains(i))
            return 1;
        else
            return 0;
    }
    // 转移特征函数
    public int t2(int y0, int y1, int[] x, int i) {
        // int(y0 in {0} and y1 in {0} and x in {(1, 1, 0), (1, 1, 1), (1, 0, 0), (1, 0, 1)} and i in {1})
        Set<Integer> Y0 = new HashSet<>();
        Set<Integer> Y1 = new HashSet<>();
        Set<Integer> IS = new HashSet<>();
        Y0.add(0);
        Y1.add(0);
        IS.add(1);

        int[][] xsdt = {{1, 1, 0}, {1, 1, 1}, {1, 0, 0}, {1, 0, 1}};
        Set<String> XS = new HashSet<>();
        for (int j = 0; j < xsdt.length; j++)
            XS.add(Arrays.toString(xsdt[j]));
        if (Y0.contains(y0) && Y1.contains(y1) && XS.contains(Arrays.toString(x)) && IS.contains(i))
            return 1;
        else
            return 0;
    }
    // 转移特征函数
    public int t3(int y0, int y1, int[] x, int i) {
        // int(y0 in {1} and y1 in {0, 1} and x in {(0, 0, 0), (1, 1, 1)} and i in {2})
        Set<Integer> Y0 = new HashSet<>();
        Set<Integer> Y1 = new HashSet<>();
        Set<Integer> IS = new HashSet<>();
        Y0.add(1);
        Y1.add(0);
        Y1.add(1);
        IS.add(2);

        int[][] xsdt = {{0, 0, 0}, {1, 1, 1}};
        Set<String> XS = new HashSet<>();
        for (int j = 0; j < xsdt.length; j++)
            XS.add(Arrays.toString(xsdt[j]));
        if (Y0.contains(y0) && Y1.contains(y1) && XS.contains(Arrays.toString(x)) && IS.contains(i))
            return 1;
        else
            return 0;
    }
    // 转移特征函数
    public int t4(int y0, int y1, int[] x, int i) {
        // int(y0 in {1} and y1 in {1} and x in {(0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1),
        //                                                     (1, 0, 0), (1, 0, 1)} and i in {2})
        Set<Integer> Y0 = new HashSet<>();
        Set<Integer> Y1 = new HashSet<>();
        Set<Integer> IS = new HashSet<>();
        Y0.add(1);
        Y1.add(1);
        IS.add(2);

        int[][] xsdt = {{0, 1, 0}, {0, 1, 1}, {0, 0, 0}, {0, 0, 1}, {1, 1, 0}, {1, 1, 1},
                {1, 0, 0}, {1, 0, 1}};
        Set<String> XS = new HashSet<>();
        for (int j = 0; j < xsdt.length; j++)
            XS.add(Arrays.toString(xsdt[j]));
        if (Y0.contains(y0) && Y1.contains(y1) && XS.contains(Arrays.toString(x)) && IS.contains(i))
            return 1;
        else
            return 0;
    }
    // 转移特征函数
    public int t5(int y0, int y1, int[] x, int i) {
        // nt(y0 in {0, 1} and y1 in {0} and x in {(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)} and i in {1, 2})
        Set<Integer> Y0 = new HashSet<>();
        Set<Integer> Y1 = new HashSet<>();
        Set<Integer> IS = new HashSet<>();
        Y0.add(0);
        Y0.add(1);
        Y1.add(0);
        IS.add(1);
        IS.add(2);

        int[][] xsdt = {{0, 1, 0}, {0, 1, 1}, {1, 1, 0}, {1, 1, 1}};
        Set<String> XS = new HashSet<>();
        for (int j = 0; j < xsdt.length; j++)
            XS.add(Arrays.toString(xsdt[j]));
        if (Y0.contains(y0) && Y1.contains(y1) && XS.contains(Arrays.toString(x)) && IS.contains(i))
            return 1;
        else
            return 0;
    }
    // 状态特征函数
    public int s1(int y0, int [] x, int i) {
        //int(y0 in {0} and x in {(0, 1, 1), (1, 1, 0), (1, 0, 1)} and i in {0, 1, 2})
        Set<Integer> Y0 = new HashSet<>();
        Set<Integer> IS = new HashSet<>();
        Y0.add(0);
        IS.add(0);
        IS.add(1);
        IS.add(2);

        int[][] xsdt = {{0, 1, 1}, {1, 1, 0}, {1, 0, 1}};
        Set<String> XS = new HashSet<>();
        for (int j = 0; j < xsdt.length; j++)
            XS.add(Arrays.toString(xsdt[j]));
        if (Y0.contains(y0) && XS.contains(Arrays.toString(x)) && IS.contains(i))
            return 1;
        else
            return 0;
    }
    // 状态特征函数
    public int s2(int y0, int [] x, int i) {
        //int(y0 in {1} and x in {(0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), (1, 0, 0),
        //                                       (1, 0, 1)} and i in {0})
        Set<Integer> Y0 = new HashSet<>();
        Set<Integer> IS = new HashSet<>();
        Y0.add(1);
        IS.add(0);

        int[][] xsdt = {{0, 1, 0}, {0, 1, 1}, {0, 0, 0},
                {0, 0, 1}, {1, 1, 0}, {1, 1, 1}, {1, 0, 0}, {1, 0, 1}};

        Set<String> XS = new HashSet<>();
        for (int j = 0; j < xsdt.length; j++)
            XS.add(Arrays.toString(xsdt[j]));
        if (Y0.contains(y0) && XS.contains(Arrays.toString(x)) && IS.contains(i))
            return 1;
        else
            return 0;
    }
    // 状态特征函数
    public int s3(int y0, int [] x, int i) {
        //int(y0 in {0} and x in {(0, 1, 1), (1, 1, 0), (1, 0, 1)} and i in {0, 1})
        Set<Integer> Y0 = new HashSet<>();
        Set<Integer> IS = new HashSet<>();
        Y0.add(0);
        IS.add(0);
        IS.add(1);

        int[][] xsdt = {{0, 1, 1}, {1, 1, 0}, {1, 0, 1}};

        Set<String> XS = new HashSet<>();
        for (int j = 0; j < xsdt.length; j++)
            XS.add(Arrays.toString(xsdt[j]));
        if (Y0.contains(y0) && XS.contains(Arrays.toString(x)) && IS.contains(i))
            return 1;
        else
            return 0;
    }
    // 状态特征函数
    public int s4(int y0, int [] x, int i) {
        //int(y0 in {1} and x in {(1, 0, 1), (0, 1, 0)} and i in {0, 2})
        Set<Integer> Y0 = new HashSet<>();
        Set<Integer> IS = new HashSet<>();
        Y0.add(1);
        IS.add(0);
        IS.add(2);

        int[][] xsdt = {{1, 0, 1}, {0, 1, 0}};

        Set<String> XS = new HashSet<>();
        for (int j = 0; j < xsdt.length; j++)
            XS.add(Arrays.toString(xsdt[j]));
        if (Y0.contains(y0) && XS.contains(Arrays.toString(x)) && IS.contains(i))
            return 1;
        else
            return 0;
    }

    /**
     * 已知条件随机场模型计算状态序列关于观测序列的非规范化条件概率
     *
     * @param w1: 模型的转移特征权重
     * @param w2: 模型的状态特征权重
     * @param x:  需要计算的观测序列
     * @param y:  需要计算的状态序列
     * @return: 状态序列关于观测序列的条件概率
     */

    public  double count_conditional_probability(double [] w1, double [] w2, int [] x, int [] y) {

        int n_features_1 = w1.length;   // 转移特征数
        int n_features_2 = w2.length;   // 状态特征数
        int n_position = x.length;      // 序列中的位置数

        double res = 0;
        for( int  k : range(n_features_1) ) {
            for( int i : range(1, n_position) ) {
                switch (k) {
                    case 0 -> res += w1[k] * t1(y[i - 1], y[i], x, i);
                    case 1 -> res += w1[k] * t2(y[i - 1], y[i], x, i);
                    case 2 -> res += w1[k] * t3(y[i - 1], y[i], x, i);
                    case 3 -> res += w1[k] * t4(y[i - 1], y[i], x, i);
                    case 4 -> res += w1[k] * t5(y[i - 1], y[i], x, i);
                    default -> {
                    }
                }
            }
        }

        for(int k : range(n_features_2) ) {
            for(int i : range(n_position) ) {
                switch (k) {
                    case 0 -> res += w2[k] * s1(y[i], x, i);
                    case 1 -> res += w2[k] * s2(y[i], x, i);
                    case 2 -> res += w2[k] * s3(y[i], x, i);
                    case 3 -> res += w2[k] * s4(y[i], x, i);
                    default -> {
                    }
                }
            }
        }
        return Math.pow(Math.E, res);
    }

    /**
     * 已知模型构造随机样本集
     *
     *     @param w1: 模型的转移特征权重
     *     @param w2: 模型的状态特征权重
     *     @param x_range: 观测序列的可能取值
     *     @param y_range: 状态序列的可能取值
     *     @param n_samples: 生成样本集样本数(近似)
     *     @return: 状态序列关于观测序列的条件概率
     */
    public ArrayList<DataFrame> make_hidden_sequence(double [] w1, double [] w2, int [][] x_range,
                                                     int [][] y_range, int n_samples, int random_state) {
        if( n_samples < 1 )
            n_samples = 1000;

        if( random_state < 0 )
            random_state = 0;

        double [][] P = new double[x_range.length][y_range.length];
        for( int i : range(x_range.length) )
            Arrays.fill(P[i], 0.0);

        ArrayList<Double> lst = new ArrayList<>();
        double sum_ = 0.0;
        for( int r : range(x_range.length) ) {
            for( int c : range(y_range.length) ) {
                P[r][c] = roundAvoid(count_conditional_probability(w1, w2, x_range[r], y_range[c]), 1);
                sum_ += P[r][c];
                lst.add(sum_);
                System.out.println("sum_ " + sum_);
            }
        }

        DataFrame X = null, Y = null;
        Random rdg = new Random(random_state);
        for( int t : range(n_samples) ) {
            double r = rdg.nextDouble(0, sum_);
            int idx = bisectLeft(lst.stream().mapToDouble(i -> i).toArray(), r);
            //i, j = divmod(idx, len(y_range))
            int i = (int)(idx *1.0 / y_range.length);
            int j = idx % y_range.length;
            int [][] x = new int[1][x_range[0].length];
            x[0] = x_range[i];
            int [][] y = new int[1][y_range[0].length];
            y[0] = y_range[j];

            if( X == null ) X = DataFrame.of(x);
            else {
                DataFrame xi = DataFrame.of(x);
                X = X.union(xi);
            }

            if( Y == null ) Y = DataFrame.of(y);
            else {
                DataFrame yi = DataFrame.of(y);
                Y = Y.union(yi);
            }
        }

        ArrayList<DataFrame> data = new ArrayList<>();
        data.add(X);
        data.add(Y);
        return data;
    }
}
