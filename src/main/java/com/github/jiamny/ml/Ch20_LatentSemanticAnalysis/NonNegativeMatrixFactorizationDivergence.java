package com.github.jiamny.ml.Ch20_LatentSemanticAnalysis;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;

import static com.github.jiamny.ml.utils.DataFrameHelper.range;

public class NonNegativeMatrixFactorizationDivergence {

    public double score(NDArray X, NDArray W, NDArray H, int n_features, int n_samples) {
        // 计算散度的损失函数
        NDArray Y = W.matMul(H);    //np.dot(W,H)
        double score = 0;
        for( int i : range(n_features) ) {
            for(int j : range(n_samples) ) {
                //score += (X[i][j] * np.log(X[i][j] / Y[i][j]) if X[i][j] * Y[i][j] > 0 else 0)+(-X[i][j] + Y[i][j])
                if( X.getDouble(i, j) * Y.getDouble(i, j) > 0 ) {
                    score += ((X.getDouble(i, j) * Math.log(X.getDouble(i, j))) +
                            (-1*X.getDouble(i, j) + Y.getDouble(i, j)));
                } else {
                    score += (0 + (-1*X.getDouble(i, j) + Y.getDouble(i, j)));
                }
            }
        }
        return score;
    }

    /**
     非负矩阵分解的迭代算法（平方损失）

     @param X: 单词-文本矩阵
     @param k: 文本集合的话题个数k
     @param max_iter: 最大迭代次数
     @param tol: 容差
     @return: 话题矩阵W,文本表示矩阵H
     */
    public ArrayList<NDArray> nmp_training(NDArray X, int k, int max_iter,
                                           double tol) {
        NDManager manager = NDManager.newBaseManager();
        int n_features = (int) (X.getShape().getShape()[0]), n_samples = (int) (X.getShape().getShape()[1]);

        // 初始化
        NDArray W = manager.randomUniform(0.0f, 1.0f, new Shape(n_features, k), DataType.FLOAT64);
        NDArray H = manager.randomUniform(0.0f, 1.0f, new Shape(k, n_samples), DataType.FLOAT64);
        System.out.println("W:\n" + W);
        System.out.println("H:\n" + H);

        // 计算当前损失函数
        double last_score = score(X, W, H, n_features, n_samples);

        // 迭代
        for( int n : range(max_iter) ) {
            // 更新W的元素
            NDArray WH = W.matMul(H);   //np.dot(W, H)
            for( int i : range(n_features) ) {
                for( int l : range(k) ) {
                    double v1 = 0;
                    for( int j : range(n_samples) )
                        v1 += (H.getDouble(l, j) * X.getDouble(i, j) / WH.getDouble(i,j));
                    //v1 = sum(H[l][j] * X[i][j] / WH[i][j] for j in range(n_samples))
                    //v2 = sum(H[l][j] for j in range(n_samples))
                    double v2 = 0;
                    for( int j : range(n_samples) )
                        v2 += H.getDouble(l, j);
                    double v = W.getDouble(i, l);
                    v *= v1 / v2;
                    W.set(new NDIndex(i, l), v);
                    //W[i][l] *= v1 / v2
                }
            }

            // 更新H的元素
            WH = W.matMul(H);        //WH = np.dot(W, H)
            for( int l : range(k) ) {
                for( int j : range(n_samples) ) {
                    //v1 = sum(W[i][l] * X[i][j] / WH[i][j] for i in range(n_features))
                    //v2 = sum(W[i][l] for i in range(n_features))
                    //H[l][j] *= v1 / v2
                    double v1 = 0;
                    for( int i : range(n_features) )
                        v1 += (W.getDouble(i, l) * X.getDouble(i, j) / WH.getDouble(i, j));
                    double v2 = 0;
                    for( int i : range(n_features) )
                        v2 += W.getDouble(i, l);
                    double v = H.getDouble(l, j);
                    v *= v1 / v2;
                    H.set(new NDIndex(l, j), v);
                }
            }

            double now_score = score(X, W, H, n_features, n_samples);
            if( last_score - now_score < tol )
                break;

            last_score = now_score;
        }
        ArrayList<NDArray> res = new ArrayList<>();
        res.add(W);
        res.add(H);
        return res;
    }

    public static void main(String[] args) {
        NDManager manager = NDManager.newBaseManager();
        NonNegativeMatrixFactorizationDivergence nnmfd = new NonNegativeMatrixFactorizationDivergence();
        double[][] Xd = {
                {0, 0, 1, 1, 0, 0, 0, 0, 0},
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
        NDArray X = manager.create(Xd).toType(DataType.FLOAT64, false);

        int max_iter = 100;
        double tol = 1e-4;

        ArrayList<NDArray> res = nnmfd.nmp_training(X, 3,  max_iter, tol);
        NDArray W = res.get(0), H = res.get(1);
        System.out.println("W:\n" + W);
        System.out.println("H:\n" + H);

        NDArray Y = W.matMul(H);
        System.out.println("Y:\n" + Y);

        double score = 0;
        int s1 = (int)X.getShape().getShape()[0], s2 = (int)X.getShape().getShape()[1];

        for(int i : range(s1)) {
            for(int j : range(s2)) {
                //score += (X[i][j] * np.log(X[i][j] / Y[i][j]) if X[i][j] * Y[i][j] > 0 else 0)+(-X[i][j] + Y[i][j])
                if( X.getDouble(i, j) * Y.getDouble(i, j) > 0 ) {
                    score += ((X.getDouble(i, j) * Math.log(X.getDouble(i, j))) +
                            (-1*X.getDouble(i, j) + Y.getDouble(i, j)));
                } else {
                    score += (0 + (-1*X.getDouble(i, j) + Y.getDouble(i, j)));
                }
            }
        }

        System.out.println("score: " + score);
        System.exit(0);
    }
}
