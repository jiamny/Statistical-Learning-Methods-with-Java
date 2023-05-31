package com.github.jiamny.ml.Ch20_LatentSemanticAnalysis;

import smile.math.matrix.Matrix;

import java.util.ArrayList;

import static com.github.jiamny.ml.utils.DataFrameHelper.range;

public class LatentSemanticAnalysisSVD {

    /**
     * 利用矩阵奇异值分解的潜在语义分析
     *
     *     @param X: 单词文本矩阵
     *     @param k: 目标话题数量
     *     @return: 话题向量空间, 文本集合在话题向量空间的表示
     */
    public ArrayList<Matrix> lsa_by_svd(Matrix X, int k) {
        Matrix.SVD svd = X.svd();
        double [] S = new double[k];
        for(int i : range(k))
            S[i] = svd.s[i];

        Matrix U = svd.U.submatrix(0, 0, svd.U.nrow()-1, k-1);
        Matrix V = svd.V.submatrix(0, 0, k-1, svd.V.ncol()-1);//[:k, :]
        ArrayList<Matrix> res = new ArrayList<>();
        res.add(U);
        res.add((Matrix.diag(S)).mm(V));
        return res;
    }

    public static void main(String[] args) {
        LatentSemanticAnalysisSVD lsa = new LatentSemanticAnalysisSVD();

        double [][] Xd = {{0, 0, 1, 1, 0, 0, 0, 0, 0},
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

        Matrix X = Matrix.of(Xd);

        ArrayList<Matrix> res = lsa.lsa_by_svd(X, 3);
        Matrix U = res.get(0), SV = res.get(1);

        System.out.println("U:\n" + U);
        System.out.println("SV:\n" + SV);
        System.exit(0);
    }
}
