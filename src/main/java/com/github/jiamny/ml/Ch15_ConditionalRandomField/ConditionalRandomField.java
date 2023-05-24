package com.github.jiamny.ml.Ch15_ConditionalRandomField;

import smile.data.DataFrame;
import smile.math.Random;
import smile.math.matrix.Matrix;

import java.util.*;

import static com.github.jiamny.ml.utils.DataFrameHelper.*;

public class ConditionalRandomField {
    private static final ProbabilisticModel PM = new ProbabilisticModel();
    public static void main(String[] args) {
        Long start = System.currentTimeMillis();

        // ---------- 《统计学习方法》例11.4 ----------
        double [] w1 = {1, 0.6, 1.2, 0.2, 1.4};
        double [] w2 = {1, 0.2, 0.8, 0.5};

        int [][] Xs = {{0, 1, 0}, {0, 1, 1}, {0, 0, 0}, {0, 0, 1}, {1, 1, 0}, {1, 1, 1}, {1, 0, 0}, {1, 0, 1}};
        int [][] Ys = {{0, 1, 0}, {0, 1, 1}, {0, 0, 0}, {0, 0, 1}, {1, 1, 0}, {1, 1, 1}, {1, 0, 0}, {1, 0, 1}};
        for( int xi : range(Xs.length) ) {
            for( int yj : range(Ys.length) ) {
                System.out.println(Arrays.toString(Xs[xi]) + " -> " + Arrays.toString(Ys[yj]) + ": " +
                        PM.count_conditional_probability(w1, w2, Xs[xi], Ys[yj]));
            }
        }

        // 生成随机模型
        int [][] x_range = {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
        int [][] y_range = {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};

        ArrayList<DataFrame> hsq = PM.make_hidden_sequence(w1, w2,x_range, y_range, 1000, 123);
        DataFrame X = hsq.get(0);
        DataFrame Y = hsq.get(1);
        for(int i : range(X.nrow())) {
            System.out.println("(" + Arrays.toString(doubleToInt(X.of(i).toMatrix().toArray()[0])) +
                    "), (" + Arrays.toString(doubleToInt(Y.of(i).toMatrix().toArray()[0])) + ")");
        }

        //double tol=1e-4, distance=20;
        //int max_iter=100;


        System.exit(0);
    }
}