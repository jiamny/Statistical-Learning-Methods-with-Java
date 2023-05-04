package com.github.jiamny.ml.Ch10_LogisticRegressionAndMaximumEntropy;

import com.github.jiamny.ml.utils.DataFrameHelper;
import org.apache.commons.csv.CSVFormat;
import smile.data.DataFrame;
import smile.io.Read;
import smile.math.MathEx;
import smile.math.matrix.Matrix;

import java.util.ArrayList;
import java.util.Arrays;

public class LogisticRegression {

    public void loadData(String fileName, ArrayList<DataFrame> tdt) {
        try {
            var format = CSVFormat.newFormat(',');
            DataFrame mnist_train = Read.csv(fileName, format);

            int [] label_idx = new int[1];
            label_idx[0] = 0;
            DataFrame train_labels = mnist_train.select(label_idx);

            // Mnsit有0-9是个标记，由于是二分类任务，所以将标记0的作为1，其余为0
            // 验证过<5为1 >5为0时正确率在90%左右，猜测是因为数多了以后，可能不同数的特征较乱，不能有效地计算出一个合理的超平面
            // 查看了一下之前感知机的结果，以5为分界时正确率81，重新修改为0和其余数时正确率98.91%
            // 看来如果样本标签比较杂的话，对于是否能有效地划分超平面确实存在很大影响
            double [][] L = train_labels.toMatrix().toArray();
            for( int r = 0; r < train_labels.nrow(); r++ )
                if( train_labels.get(r, 0) == Integer.valueOf(0) )
                    L[r][0] = 1;
                else
                    L[r][0] = 0;
            train_labels = DataFrame.of(L);


            int [] data_idx = new int[mnist_train.ncol()-1];
            for(int i = 1; i <= (mnist_train.ncol()-1); i++ )
                data_idx[i - 1] = i;
            DataFrame train_data = mnist_train.select(data_idx);

            // 将所有数据除255归一化(非必须步骤，可以不归一化
            double [][] mdt = new double[train_data.nrow()][train_data.ncol()];

            for(int r = 0; r < train_data.nrow(); r++) {
                for(int c = 0; c < train_data.ncol(); c++) {
                    mdt[r][c] = train_data.getDouble(r, c)/255;
                }
            }
            DataFrame ntrain_data = DataFrame.of(mdt);

            tdt.add(ntrain_data);
            tdt.add(train_labels);
        } catch(Exception e) {
            e.printStackTrace();
        }
    }

    /**
     逻辑斯蒂回归训练过程
     @param trainDataList:训练集
     @param trainLabelList: 标签集
     @param iter: 迭代次数
     @return: 习得的w
     */
    public Matrix logisticRegression( DataFrame trainDataList, DataFrame trainLabelList, int iter ) {

        if( iter < 200 )
            iter = 200;

        // 按照书本“6.1.2 二项逻辑斯蒂回归模型”中式6.5的规则，将w与b合在一起，
        // 此时x也需要添加一维，数值为1
        // 循环遍历每一个样本，并在其最后添加一个1
        double [][] append = new double[trainDataList.nrow()][1];
        for( int k = 0; k < append.length; k++ )
            append[k][0] = 1.0;
        trainDataList = trainDataList.merge(DataFrame.of(append, "V" + String.valueOf(trainDataList.ncol()+1)));

        // 将数据集由列表转换为数组形式，主要是后期涉及到向量的运算，统一转换成数组形式比较方便
        //trainDataList = np.array(trainDataList)
        Matrix trdata = trainDataList.toMatrix();
        Matrix tlabels = trainLabelList.toMatrix();

        // 初始化w，维数为样本x维数+1，+1的那一位是b，初始为0
        // w = np.zeros(trainDataList.shape[1])

        double [][] wd = new double[1][trdata.ncol()];
        int [] cols = new int[trdata.ncol()];
        for(int k = 0; k < trdata.ncol(); k++ ) {
            wd[0][k] = 0.0;
            cols[k] = k;
        }
        Matrix w = Matrix.of(wd);

        int [] row = new int[1];
        int [] col = new int[1];
        col[0] = 0;

        // 设置步长
        double h = 0.001;

        // 迭代iter次进行随机梯度下降
        for(int i = 0; i < iter; i++ ) {
            // 每次迭代冲遍历一次所有样本，进行随机梯度下降
            for (int j = 0; j < trdata.nrow(); j++) {
                // 随机梯度上升部分
                // 在“6.1.3 模型参数估计”一章中给出了似然函数，我们需要极大化似然函数
                // 但是似然函数由于有求和项，并不能直接对w求导得出最优w，所以针对似然函数求和
                // 部分中每一项进行单独地求导w，得到针对该样本的梯度，并进行梯度上升（因为是
                // 要求似然函数的极大值，所以是梯度上升，如果是极小值就梯度下降。梯度上升是加号，下降是减号）
                // 求和式中每一项单独对w求导结果为：xi * yi - (exp(w * xi) * xi) / (1 + exp(w * xi))
                // 如果对于该求导式有疑问可查看我的博客 www.pkudodo.com

                // 计算w * xi，因为后式中要计算两次该值，为了节约时间这里提前算出
                // 其实也可直接算出exp(wx)，为了读者能看得方便一点就这么写了，包括yi和xi都提前列出了

                double wx = MathEx.dot(w.row(0), trdata.row(j));
                row[0] = j;

                double yi = tlabels.get(j, 0);
                Matrix xi = trdata.get(row, cols);
                double wxe = Math.exp(wx);

                Matrix t1 = (xi.clone().mul(wxe)).div(wxe + 1.0);
                //System.out.println("t1: " + t1.sum());

                Matrix t2 = ((xi.mul(yi)).sub(t1)).mul(h);
                //System.out.println("t2: " + t2.sum());
                w = w.add(t2);

                //w += h * (xi * yi - (np.exp(wx) * xi) / (1 + np.exp(wx)));
            }
            System.out.println("iter: " + i + " w: " + w.sum());
        }
        // 返回学到的w
        return w;
    }

    /**
     * 预测标签
     * @param w:训练过程中学到的w
     * @param x: 要预测的样本
     * @return: 预测结果
     */
    public int predict(Matrix w, double[] x) {

        // dot为两个向量的点积操作，计算得到w * x
        double wx = MathEx.dot( w.row(0), x);
        // 计算标签为1的概率
        // 该公式参考“6.1 .2 二项逻辑斯蒂回归模型”中的式6 .5
        double P1 = Math.exp(wx) / (1.0 + Math.exp(wx));

        // 如果为1的概率大于0 .5，返回1
        if( P1 >= 0.5 )
            return 1;
        // 否则返回0
        else
            return 0;
    }

    /**
     验证
     @param testDataList:测试集
     @param testLabelList: 测试集标签
     @param w: 训练过程中学到的w
     @return: 正确率
     */
    public double model_test(DataFrame testDataList, DataFrame testLabelList, Matrix w) {
        // 与训练过程一致，先将所有的样本添加一维，值为1，理由请查看训练函数
        double [][] append = new double[testDataList.nrow()][1];
        for( int k = 0; k < append.length; k++ )
            append[k][0] = 1.0;
        testDataList = testDataList.merge(DataFrame.of(append, "V" + String.valueOf(testDataList.ncol()+1)));

        Matrix trdata = testDataList.toMatrix();
        // 错误值计数
        int errorCnt = 0;
        // 对于测试集中每一个测试样本进行验证
        //int [] labels = Arrays.stream(testLabelList.toMatrix().row(0)).mapToInt(
        //        i -> (int) i).toArray();

        for(int i = 0; i < trdata.nrow(); i++ ) {
            // 如果标记与预测不一致，错误值加1
            if( ((int)testLabelList.getDouble(i, 0)) != predict(w, trdata.row(i)))
                errorCnt += 1;
        }
        // 返回准确率
        System.out.println("errorCnt: " + errorCnt + " tot: " + trdata.nrow());
        return (1.0 - (errorCnt * 1.0 / trdata.nrow()));
    }
    public static void main(String [] args) {
        LogisticRegression LR = new LogisticRegression();
        String fnm = "./data/Mnist/mnist_train.csv";
        ArrayList<DataFrame> train_data = new ArrayList<>();
        LR.loadData(fnm, train_data);

        DataFrame trainLabelList = train_data.get(1);
        DataFrame trainDataList = train_data.get(0);
        //System.out.println("trainDataList: " + trainDataList.of(0));
        Matrix w = LR.logisticRegression(trainDataList, trainLabelList, 2);

        fnm = "./data/Mnist/mnist_test.csv";
        ArrayList<DataFrame> test_data = new ArrayList<>();
        LR.loadData(fnm, test_data);

        System.out.println("The accur is: " + LR.model_test(test_data.get(0), test_data.get(1), w));

        System.exit(0);
    }
}
