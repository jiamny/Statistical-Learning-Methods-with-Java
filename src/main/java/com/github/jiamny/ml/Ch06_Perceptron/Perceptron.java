package com.github.jiamny.ml.Ch06_Perceptron;

import com.github.jiamny.ml.utils.ImageViewer;
import org.apache.commons.csv.CSVFormat;
import smile.data.DataFrame;
import smile.data.vector.IntVector;
import smile.io.Read;
import smile.math.matrix.Matrix;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;

import static com.github.jiamny.ml.utils.StatisticHelper.printVectorElements;

class PerceptronParameters {
    private Matrix weights = null;
    private double bias = 0.0;

    public double getBias() {
        return bias;
    }

    public Matrix getWeights() {
        return weights;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public void setWeights(Matrix weights) {
        this.weights = weights;
    }
}
public class Perceptron {
    public ArrayList<DataFrame> loadData (String trainName, String testName) {

        ArrayList<DataFrame> rlt = new ArrayList<>();
        try {
            var format = CSVFormat.newFormat(',');
            var mnist_train = Read.csv(trainName, format);
            var mnist_test = Read.csv(testName, format);
            rlt.add(mnist_train);
            rlt.add(mnist_test);
        } catch(Exception e) {
            e.printStackTrace();
        }
        return rlt;
    }

    PerceptronParameters perceptron(DataFrame train_data, IntVector trainLabels,Matrix data,
                                                    PerceptronParameters param, int num_iter) {

        Matrix ww = param.getWeights();
        double b = param.getBias();
        // Mnsit有0-9是个标记，由于是二分类任务，所以将>=5的作为1，<5为-1
        int [] trlabels = trainLabels.toIntArray();
        for( int i = 0; i < trlabels.length; i++ ) {
            if( trlabels[i] >= 5 )
                trlabels[i] = 1;
            else
                trlabels[i] = -1;
        }
        printVectorElements(trlabels);

        int m = train_data.nrow();

        int iter = num_iter;
        // 初始化步长，也就是梯度下降过程中的n，控制梯度下降速率
        double hs = 0.0001;
        // 进行iter次迭代计算
        System.out.println("+++1");
        for( int k = 0; k < iter; k++ ) {

            for( int j = 0; j < m; j ++ ) {
                var yi = trainLabels.get(j);
                var xi = data.col(j);
                var a = ww.mv(xi);

                if( -1 * yi * (a[0] + b) >= 0.0 ) {
                    //对于误分类样本，进行梯度下降，更新w和b
                    //w = w + hs * yi * xi;
                    for( int i = 0; i < ww.ncol(); i++ )
                        ww.set(0, i, hs * yi * xi[i] );
                    b = b + hs * yi;
                }
            }
            System.out.println("Round " + (k + 1) + "/" + iter + " training" );
        }
        param.setBias(b);
        param.setWeights(ww);

        return param;
    }

    public void displayImage( DataFrame train_data, int imgIdx, int imgSize) {
        int w = 28, h = 28;
        try {
            BufferedImage image = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
            int k = 0;
            for(int i=0; i < w; i++) {
                for(int j=0; j< h; j++) {
                    int a = train_data.getInt(imgIdx, k);
                    Color newColor = new Color(a,a,a);
                    image.setRGB(j,i,newColor.getRGB());
                    k++;
                }
            }
            // resize image
            BufferedImage nimage = ImageViewer.getScaledImage(image, imgSize, imgSize);
            ImageViewer.display(nimage);
        } catch(Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    double model_test(DataFrame test_data, IntVector testLabels, PerceptronParameters param) {
        //获取测试数据集矩阵的大小
        int m = test_data.nrow();
        int n = test_data.ncol();
        Matrix w = param.getWeights();
        double b = param.getBias();
        //错误样本数计数
        int errorCnt = 0;
        Matrix tstdata = test_data.toMatrix().transpose();

        // 遍历所有测试样本
        for(int i = 0; i < m; i++) {
            //获得单个样本向量
            var xi = tstdata.col(i);
            //获得该样本标记
            var yi = testLabels.get(i);
            //获得运算结果
            var result = -1 * yi * (w.mv(xi)[0] + b);
            // 如果-yi(w*xi+b)>=0，说明该样本被误分类，错误样本数加一
            if(result >= 0)
                errorCnt += 1;
        }
        //正确率 = 1 - （样本分类错误数 / 样本总数）
        double accruRate = 1 - (errorCnt*1.0 / m);
        //返回正确率
        return accruRate;
    }


    public static void main( String [] args ) {
        Perceptron pp = new Perceptron();

        ArrayList<DataFrame> mnistData = pp.loadData("data/Mnist/mnist_train.csv", "data/Mnist/mnist_test.csv");
        DataFrame train_data_labels = mnistData.get(0);
        DataFrame test_data_labels = mnistData.get(1);

        IntVector trainLabels = train_data_labels.intVector(0);
        System.out.println(trainLabels.size());
        System.out.println(train_data_labels.drop(0).ncol());

        DataFrame train_data = train_data_labels.drop(0);

        // show image 5
        pp.displayImage(train_data, 4, 280);

        int n = train_data.ncol();
        Matrix data = train_data.toMatrix().transpose();
        double [][] dw  = new double[1][n];
        for(int r = 0; r < n; r++)
            dw[0][r] = 0.0;
        Matrix ww = DataFrame.of(dw).toMatrix();

        PerceptronParameters param = new PerceptronParameters();
        param.setWeights(ww);
        param.setBias(0.0);
        param = pp.perceptron(train_data, trainLabels, data, param, 50);

        System.out.println(param.getWeights());
        System.out.println(param.getBias());

        IntVector testLabels = test_data_labels.intVector(0);
        System.out.println(testLabels.size());

        DataFrame test_data = test_data_labels.drop(0);

        double accruRate = pp.model_test(test_data, testLabels, param);
        System.out.println("正确率: " + accruRate);

        System.exit(0);
    }
}
