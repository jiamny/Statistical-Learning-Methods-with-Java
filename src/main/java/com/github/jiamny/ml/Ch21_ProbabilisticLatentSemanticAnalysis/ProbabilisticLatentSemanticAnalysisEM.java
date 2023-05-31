package com.github.jiamny.ml.Ch21_ProbabilisticLatentSemanticAnalysis;

import smile.math.Random;
import smile.math.matrix.Matrix;

import java.util.ArrayList;

import static com.github.jiamny.ml.utils.DataFrameHelper.range;

public class ProbabilisticLatentSemanticAnalysisEM {

    /**
     * 概率潜在语义模型参数估计的EM算法
     *
     *     @param X: 单词-文本共现矩阵
     *     @param K: 话题数量
     *     @param max_iter: 最大迭代次数
     *     @param random_state: 随机种子
     *     @return: P(w_i|z_k)和P(z_k|d_j)
     */
    public ArrayList<Matrix> em_for_plsa(double [][] X, int K, int max_iter, long random_state) {
        int n_features = X.length, n_samples = X[0].length;

        // 计算n(d_j)
        //N = [np.sum(X[:, j]) for j in range(n_samples)]
        double [] N = new double[n_samples];
        for( int j : range(n_samples) ) {
            double nv = 0;
            for( int i : range(n_features) )
                nv += X[i][j];
            N[j] = nv;
        }

        Random rg = new Random(random_state);

        // 设置参数P(w_i|z_k)和P(z_k|d_j)的初始值
        double [][] P1 = new double[n_features][K];  // P(w_i|z_k)
        double [][] P2 = new double[K][n_samples];   // P(z_k|d_j)

        for(int r : range(n_features)) {
            for(int c : range(K))
                P1[r][c] = rg.nextDouble(0.0, 1.0);
        }

        for(int r : range(K)) {
            for(int c : range(n_samples))
                P2[r][c] = rg.nextDouble(0.0, 1.0);
        }

        for( int l : range(max_iter) ) {
            //E步
            double [][][] P = new double[n_features][n_samples][K];
            for(int i : range(n_features)) {
                for(int j : range(n_samples)) {
                    double ksum = 0;
                    for(int k : range(K)) {
                        P[i][j][k] = P1[i][k] * P2[k][j];
                        ksum += P1[i][k] * P2[k][j];
                    }
                    //P[i][j] /= np.sum(P[i][j])
                    for(int k : range(K))
                        P[i][j][k] /= ksum;
                }
            }

            // M步
            for(int k : range(K)) {
                double psum = 0;
                for(int i : range(n_features)) {
                    double v = 0;
                    for(int j : range(n_samples) )
                        v += (X[i][j] * P[i][j][k]);
                    P1[i][k] = v;
                    psum += v;
                }
                //P1[:,k] /=np.sum(P1[:,k])
                for(int j : range(n_features))
                    P1[j][k] /= psum;
            }
            for( int  k : range(K) ) {
                for(int j : range(n_samples)) {
                    double tsum = 0;
                    for(int i : range(n_features))
                        tsum += (X[i][j] * P[i][j][k]);
                    P2[k][j] = tsum /N[j];
                }
            }
        }
        ArrayList<Matrix> res = new ArrayList<>();
        res.add(Matrix.of(P1));
        res.add(Matrix.of(P2));
        return res;
    }

    public static void main(String[] args) {
        ProbabilisticLatentSemanticAnalysisEM plsa = new ProbabilisticLatentSemanticAnalysisEM();
        double [][] X = {{0, 0, 1, 1, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1, 0, 0, 1},
                {0, 1, 0, 0, 0, 0, 0, 1, 0},
                {0, 0, 0, 0, 0, 0, 1, 0, 1},
                {1, 0, 0, 0, 0, 1, 0, 0, 0},
                {1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 0, 1, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 1, 0, 1},
                {0, 0, 0, 0, 0, 2, 0, 0, 1},
                {1, 0, 1, 0, 0, 0, 0, 1, 0},
                {0, 0, 0, 1, 1, 0, 0, 0, 0}};

        int max_iter = 100;
        long random_state = 0;
        ArrayList<Matrix> res = plsa.em_for_plsa(X, 3, max_iter, random_state);

        Matrix R1 = res.get(0), R2 = res.get(1);

        System.out.println("R1:\n" + R1);
        System.out.println("R2:\n" + R2);
        System.exit(0);
    }
}
