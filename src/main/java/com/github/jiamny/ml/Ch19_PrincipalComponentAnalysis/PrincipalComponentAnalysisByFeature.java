package com.github.jiamny.ml.Ch19_PrincipalComponentAnalysis;

import smile.math.matrix.Matrix;

import java.util.Arrays;

import static com.github.jiamny.ml.utils.DataFrameHelper.range;
import static com.github.jiamny.ml.utils.StatisticHelper.argSort;
import static com.github.jiamny.ml.utils.StatisticHelper.printVectorElements;

public class PrincipalComponentAnalysisByFeature {

    /**
     * 协方差矩阵/相关矩阵求解主成分及其因子载荷量和贡献率（打印到控制台）
     *
     *     @param R: 协方差矩阵/相关矩阵
     *     @param need_accumulative_contribution_rate: 需要达到的累计方差贡献率
     *     @return: None
     */
    public void pca_by_feature(Matrix R, double need_accumulative_contribution_rate) {
        int n_features = R.nrow();

        // 求解相关矩阵的特征值和特征向量
        Matrix.EVD evd = R.eigen();         // np.linalg.eig(R)
        double [] features_value = evd.wr;
        Matrix features_vector = evd.Vr;

        // 依据特征值大小排序特征值和特征向量
        /*
        z = [(features_value[i], features_vector[:, i]) for i in range(n_features)]
        z.sort(key=lambda x: x[0], reverse=True)
        features_value = [z[i][0] for i in range(n_features)]
        features_vector = np.hstack([z[i][1][:, np.newaxis] for i in range(n_features)])
        */
        double [] revlambda_V = new double[features_value.length];
        for(int i : range(revlambda_V.length))
            revlambda_V[i] = -1*features_value[i];

        int [] index = argSort(revlambda_V);
        printVectorElements(index);

        double [][] lambda_V = new double[1][features_value.length];

        double [][] sortMat = new double [features_vector.nrow()][features_vector.ncol()];
        for( int i  : index ) {
            lambda_V[0][i] = features_value[i];
            double [] col = features_vector.col(i);
            for( int j = 0; j < features_vector.nrow(); j++ )
                sortMat[j][i] = col[j];
        }
        features_vector = Matrix.of(sortMat);
        Matrix features_values = Matrix.of(lambda_V);

        // 计算所需的主成分数量

        double total_features_value = features_values.sum();  // 特征值总和
        need_accumulative_contribution_rate *= total_features_value;

        int n_principal_component = 0;  // 所需的主成分数量
        double accumulative_contribution_rate = 0;

        while( accumulative_contribution_rate < need_accumulative_contribution_rate ) {
            accumulative_contribution_rate += features_values.get(0, n_principal_component);
            n_principal_component += 1;
        }

        // 输出单位特征向量和主成分的方差贡献率
        System.out.println("【单位特征向量和主成分的方差贡献率】");
        for(int i : range(n_principal_component) ) {
            System.out.print("主成分: " + i +
                    " 方差贡献率: " + (features_values.get(0, i) / total_features_value) +
                    " 特征向量: ");
            printVectorElements(features_vector.col(i));
        }

        // 计算各个主成分的因子载荷量：factor_loadings[i][j]表示第i个主成分对第j个变量的相关关系，即因子载荷量
        double [][] factor_loadings = new double [n_principal_component][n_features];
        for(int i : range(n_principal_component)) {
            for(int j : range(n_features)) {
                factor_loadings[i][j] = Math.sqrt(features_values.get(0, i)) *
                        features_vector.get(j, i) / Math.sqrt(R.get(j, j));
            }
        }

        // 输出主成分的因子载荷量和贡献率
        System.out.println("\n【主成分的因子载荷量和贡献率】");
        for(int i : range(n_principal_component) ) {
            System.out.print("主成分: " + i + " 因子载荷量: ");
            printVectorElements(factor_loadings[i]);
        }
        System.out.print("所有主成分对变量的贡献率: "); //[np.sum(factor_loadings[:,j] **2)])
        for(int j : range(n_features) ) {
            double sum = 0;
            for (int k : range(factor_loadings.length))
                sum += Math.pow(factor_loadings[k][j], 2);
            System.out.print(sum + " ");
        }
        System.out.println();
    }

    public static void main(String[] args) {
        PrincipalComponentAnalysisByFeature pcaBF = new PrincipalComponentAnalysisByFeature();

        double [][] Xd = {{1, 0.44, 0.29, 0.33},
                {0.44, 1, 0.35, 0.32},
                {0.29, 0.35, 1, 0.60},
                {0.33, 0.32, 0.60, 1}};

        pcaBF.pca_by_feature(Matrix.of(Xd), 0.75);
    }
}
