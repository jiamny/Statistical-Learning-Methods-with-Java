package com.github.jiamny.ml.Ch19_PrincipalComponentAnalysis;

import com.github.jiamny.ml.Ch15_ConditionalRandomField.IIS_Algorithm;
import com.github.jiamny.ml.utils.MathExHelper;
import smile.data.DataFrame;
import smile.io.CSV;
import smile.math.matrix.Matrix;

import java.nio.file.Paths;
import java.util.Arrays;

import static com.github.jiamny.ml.utils.StatisticHelper.printVectorElements;

public class PrincipalComponentAnalysis {

    //定义加载数据的函数
    public DataFrame load_data(String file) {
        DataFrame iris_data = null;

        try {
            CSV rd = new CSV();
            iris_data = rd.read(Paths.get(file));
            //Arrays.stream(car_data.names()).forEach(System.out::println);
            System.out.println(iris_data.schema());
            System.out.println(iris_data.schema());
        } catch(Exception e) {
            e.printStackTrace();
        }
        return iris_data;
    }

    public static void main(String[] args) {
        PrincipalComponentAnalysis pca = new PrincipalComponentAnalysis();

        DataFrame  data = pca.load_data("data/irisX.csv");
        Matrix X = data.toMatrix();
        int n = X.nrow();
        double [] avg = X.colMeans();
        //X = X - np.mean(X, axis=0)
        for( int c = 0; c < X.ncol(); c++ ) {
            for( int r = 0; r < X.nrow(); r++ )
                X.set(r, c, X.get(r, c) - avg[c]);
        }
        Matrix XX = X.clone();
        // np.linalg.svd((X.T @ X)/n)
        Matrix t = (X.transpose().mm(X)).div(n);
        Matrix.SVD svd = t.svd();
        Matrix U = svd.U;
        System.out.println("U = \n" + U);
        System.out.print("\ndiag(D^2) = ");
        printVectorElements(svd.s);
        double [][] c1 = MathExHelper.arrayBindByCol(U.col(0));
        Matrix z =  Matrix.of(c1).transpose().mm(XX.transpose());
        System.out.println("z = \n" + z);
    }
}

