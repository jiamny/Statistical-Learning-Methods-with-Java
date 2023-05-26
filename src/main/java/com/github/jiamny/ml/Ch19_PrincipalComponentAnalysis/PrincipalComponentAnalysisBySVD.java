package com.github.jiamny.ml.Ch19_PrincipalComponentAnalysis;

import smile.math.matrix.Matrix;

import java.util.Arrays;

import static com.github.jiamny.ml.utils.DataFrameHelper.range;

public class PrincipalComponentAnalysisBySVD {
    /**
     * 数据矩阵奇异值分解进行的主成分分析算法
     *
     *     @param X: 样本矩阵X
     *     @param k: 主成分个数k
     *     @return:
     */
    public Matrix pca_by_svd(Matrix X, int k) {
        int n_samples = X.ncol();
        Matrix XX = X.clone();      // save a copy of X

        // 构造新的n×m矩阵
        Matrix T = (X.transpose()).div(Math.sqrt(n_samples - 1)); // np.sqrt(n_samples - 1)

        // 对矩阵T进行截断奇异值分解
        Matrix.SVD svd = T.svd();
        //U, S, V = np.linalg.svd(T)
        Matrix V = svd.V;
        System.out.println("V: " + V);
        V = V.submatrix(0, 0, V.nrow() -1, k - 1); //[:, :k]

        // 求k×n的样本主成分矩阵
        return (V.transpose()).mm(XX); //np.dot(V.T, X);
    }

    public static void main(String[] args) {
        PrincipalComponentAnalysisBySVD pca = new PrincipalComponentAnalysisBySVD();
        double [][] Xdt = {{2, 3, 3, 4, 5, 7},
                {2, 4, 5, 5, 6, 8}};
        Matrix X = Matrix.of(Xdt);

        // 规范化变量
        double [] avg = X.rowMeans();   //np.average(X, axis=1)
        double [] var = X.rowSds();     //np.var(X, axis=1)
        Arrays.stream(var).forEach(System.out::println);
        for( int i : range(X.nrow()) ) {
            for( int j : range(X.ncol())) {
                //X[i] = (X[i, :]-avg[i]) /var[i]
                X.set(i, j, (X.get(i, j) - avg[i])/var[i]);
            }
        }
        System.out.println("X: " + X);
        System.out.println(pca.pca_by_svd(X, 2));

        System.exit(0);
    }
}
