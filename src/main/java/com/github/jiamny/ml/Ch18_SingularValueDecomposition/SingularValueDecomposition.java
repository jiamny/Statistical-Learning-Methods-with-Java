package com.github.jiamny.ml.Ch18_SingularValueDecomposition;

import smile.math.matrix.Matrix;
import java.util.ArrayList;
import java.util.Arrays;

import static com.github.jiamny.ml.utils.DataFrameHelper.range;
import static com.github.jiamny.ml.utils.StatisticHelper.argSort;
import static com.github.jiamny.ml.utils.StatisticHelper.printVectorElements;

public class SingularValueDecomposition {

    // 基于矩阵分解的结果，复原矩阵
    public Matrix rebuildMatrix(Matrix U, Matrix sigma, Matrix V) {
        Matrix a = U.mm(sigma);     //np.dot(U, sigma)
        a = a.mm(V.transpose());    //np.dot(a, np.transpose(V))
        return a;
    }

    // 基于特征值的大小，对特征值以及特征向量进行排序。倒序排列
    public ArrayList<Matrix> sortByEigenValue(double [] Eigenvalues, Matrix EigenVectors) {
        ArrayList<Matrix> res = new ArrayList<>();
        printVectorElements(Eigenvalues);
        double [] revlambda_V = new double[Eigenvalues.length];
        for(int i : range(revlambda_V.length))
            revlambda_V[i] = -1*Eigenvalues[i];

        int [] index = argSort(revlambda_V);
        printVectorElements(index);

        double [][] lambda_V = new double[1][Eigenvalues.length];

        double [][] sortMat = new double [EigenVectors.nrow()][EigenVectors.ncol()];
        for( int i  : index ) {
            lambda_V[0][i] = Eigenvalues[i];
            double [] col = EigenVectors.col(i);
            for( int j = 0; j < EigenVectors.nrow(); j++ )
                sortMat[j][i] = col[j];
        }
        EigenVectors = Matrix.of(sortMat);
        res.add(Matrix.of(lambda_V));
        res.add(EigenVectors);
        return res;
    }

    //对一个矩阵进行奇异值分解
    public ArrayList<Matrix> SVD(Matrix matrixA, int NumOfLeft) {
        // NumOfLeft是要保留的奇异值的个数，也就是中间那个方阵的宽度
        // 首先求transpose(A)*A
        // matrixAT_matrixA = np.dot(np.transpose(matrixA), matrixA)
        Matrix matrixAT_matrixA = matrixA.transpose().mm(matrixA);
        System.out.println("matrixA: " + matrixAT_matrixA);

        //然后求右奇异向量
        //lambda_V, X_V = np.linalg.eig(matrixAT_matrixA)
        //lambda_V, X_V = sortByEigenValue(lambda_V, X_V)
        Matrix.EVD evd = matrixAT_matrixA.eigen();

        ArrayList<Matrix> res = sortByEigenValue(evd.wr, evd.Vr);
        Matrix lambda_V = res.get(0);
        Matrix X_V = res.get(1);

        //求奇异值
        ArrayList<Double> sigmas = new ArrayList<>();
        //sigmas = list(map(lambda x: np.sqrt(x)
        //if x > 0 else 0, sigmas))
        for( double x : lambda_V.row(0) ) {
            if( x < 0)
                x = 0.0;
            sigmas.add(Math.sqrt(x));
        }

        double [] sigmaA = sigmas.stream().mapToDouble(i -> i).toArray();
        Matrix sigmasMatrix = Matrix.diag(sigmaA);
        int rankOfSigmasMatrix = 0;
        if( NumOfLeft < 1 ) {
            for( double x : sigmaA )
                if( x > 0 ) rankOfSigmasMatrix++;
        } else {
            rankOfSigmasMatrix = NumOfLeft;
        }

        // 特征值为0的奇异值就不要了
        sigmasMatrix =  sigmasMatrix.submatrix(0, 0, rankOfSigmasMatrix-1, (sigmasMatrix.ncol()-1));

        // 计算右奇异向量
        // 初始化一个右奇异向量矩阵，这里直接进行裁剪
        double [][] XUdt = new double[matrixA.nrow()][rankOfSigmasMatrix];
        for(int i : range(XUdt.length))
            Arrays.fill(XUdt[i], 0.0);

        for(int i = 0; i < rankOfSigmasMatrix; i++ ) {
            //X_U[:,i] =np.transpose(np.dot(matrixA, X_V[:,i]) /sigmas[i])
            double [] t = matrixA.mv(X_V.col(i));
            for( int j = 0; j < XUdt.length; j++) {
                XUdt[j][i] = (t[j]/sigmaA[i]);
            }
        }

        // 对右奇异向量和奇异值矩阵进行裁剪
        X_V = X_V.submatrix(0, 0, X_V.nrow() - 1, NumOfLeft - 1);
        sigmasMatrix = sigmasMatrix.submatrix(0, 0, rankOfSigmasMatrix-1, rankOfSigmasMatrix-1);
        ArrayList<Matrix> svd = new ArrayList<>();
        svd.add(Matrix.of(XUdt));
        svd.add(sigmasMatrix);
        svd.add(X_V);
        return svd;
    }

    public static void main(String[] args) {
        SingularValueDecomposition svd = new SingularValueDecomposition();

        double [][] xmat = new double[][]{
                {1, 1, 1, 2, 2},
                {0, 0, 0, 3, 3},
                {0, 0, 0, 1, 1},
                {1, 1, 1, 0, 0},
                {2, 2, 2, 0, 0},
                {5, 5, 5, 0, 0},
                {1, 1, 1, 0, 0}};
        Matrix matrixA = Matrix.of(xmat);
        int NumOfLeft = 3;
        System.out.println("submatric: " + matrixA.submatrix(0, 0, matrixA.nrow()-1, 3-1));

        ArrayList<Matrix> res = svd.SVD(matrixA, NumOfLeft);
        System.out.println("X_U: " + res.get(0));
        System.out.println("sigmas: " + res.get(1));
        System.out.println("X_V: " + res.get(2));


        Matrix orgNd = svd.rebuildMatrix(res.get(0), res.get(1), res.get(2));
        System.out.println("Rebuild matrixA: " + orgNd);
    }
}
