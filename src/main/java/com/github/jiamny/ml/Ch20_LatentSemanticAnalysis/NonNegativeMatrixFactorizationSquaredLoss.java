package com.github.jiamny.ml.Ch20_LatentSemanticAnalysis;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import smile.math.Random;
import smile.math.matrix.Matrix;

import java.util.ArrayList;

import static com.github.jiamny.ml.utils.DataFrameHelper.range;

public class NonNegativeMatrixFactorizationSquaredLoss {

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
        int n_features = (int)(X.getShape().getShape()[0]), n_samples = (int)(X.getShape().getShape()[1]);

        // 初始化
        NDArray W = manager.randomUniform(0.0f, 1.0f, new Shape(n_features, k), DataType.FLOAT64);
        NDArray H = manager.randomUniform(0.0f, 1.0f, new Shape(k, n_samples), DataType.FLOAT64);
        System.out.println("W:\n" + W);
        System.out.println("H:\n" + H);

        // 计算当前平方损失
        //last_score = np.sum(np.square(X - np.dot(W, H)))
        NDArray T = X.sub(W.matMul(H));
        double last_score = T.square().sum().toDoubleArray()[0];
        System.out.println("last_score:\n" + last_score);

        // 迭代
        for( int n : range(max_iter) ) {

            // 更新W的元素
            //A = np.dot(X, H.T)  // X H^T
            NDArray A = X.matMul(H.transpose());
            //B = np.dot(np.dot(W, H), H.T)  // W H H^T
            NDArray B = (W.matMul(H)).matMul(H.transpose());
            for( int i : range(n_features) ) {
                for(int l : range(k) ) {
                    double v = W.getDouble(i, l);
                    v *= A.getDouble(i, l) / B.getDouble(i, l);
                    W.set(new NDIndex(i, l), v);
                }
            }

            // 更新H的元素
            NDArray C = (W.transpose()).matMul(X); //np.dot(W.T, X)  //# W^T X
            NDArray D = ((W.transpose()).matMul(W)).matMul(H);   //np.dot(np.dot(W.T, W), H)  # W^T W H
            for(int l : range(k) ) {
                for(int j : range(n_samples) ){
                    //H[l][j] *= C[l][j] / D[l][j]
                    double v = H.getDouble(l, j);
                    v *= C.getDouble(l, j) / D.getDouble(l, j);
                    H.set(new NDIndex(l, j),  v);
                }
            }

            // 检查迭代更新量是否已小于容差
            // now_score = np.sum(np.square(X - np.dot(W, H)))
            T = X.sub(W.matMul(H));

            double now_score = T.square().sum().toDoubleArray()[0];

            System.out.println("last_score - now_score: " + (last_score - now_score));
            if((last_score - now_score) < tol)
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
        NonNegativeMatrixFactorizationSquaredLoss nnmfsl = new NonNegativeMatrixFactorizationSquaredLoss();
        double [][] Xd = {
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

        int max_iter=100;
        double tol=1e-4;

        ArrayList<NDArray> res = nnmfsl.nmp_training(X, 3,  max_iter, tol);
        NDArray W = res.get(0), H = res.get(1);
        System.out.println("W:\n" + W);
        System.out.println("H:\n" + H);

        NDArray Y = W.matMul(H);
        System.out.println("Y:\n" + Y);

        System.out.println("square loss: \n" + X.sub(Y).square().sum().toDoubleArray()[0]);
    }
}
