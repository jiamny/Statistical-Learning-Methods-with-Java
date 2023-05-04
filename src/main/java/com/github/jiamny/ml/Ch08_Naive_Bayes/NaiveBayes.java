package com.github.jiamny.ml.Ch08_Naive_Bayes;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import com.github.jiamny.ml.utils.DataFrameHelper;
import smile.data.DataFrame;

import java.util.ArrayList;
import com.github.jiamny.ml.utils.StatisticHelper;

public class NaiveBayes {

    /**
     * 通过朴素贝叶斯进行概率估计
     * @param Py: 先验概率分布
     * @param Px_y0: 条件概率分布
     * @param Px_y1: 条件概率分布
     * @param x: 要估计的样本x
     * #return: 返回所有label的估计概率
     */
    public int naiveBayes(DataFrame Py, DataFrame Px_y0, DataFrame Px_y1, DataFrame x) {

        // 设置特征数目
        int featrueNum = 784;
        // 设置类别数目
        int classNum = 10;
        // 建立存放所有标记的估计概率数组
        double [] P = new double[classNum];
        for (int i = 0; i < classNum; i++) {
            P[i] = 0.0;
        }

        // 对于每一个类别，单独估计其概率
        for (int i = 0; i < classNum; i++) {
            // 初始化sum为0，sum为求和项。
            // 在训练过程中对概率进行了log处理，所以这里原先应当是连乘所有概率，最后比较哪个概率最大
            // 但是当使用log处理时，连乘变成了累加，所以使用sum
            double sum = 0;
            // 获取每一个条件概率值，进行累加
            for( int j = 0; j < featrueNum; j++ ) {
                //NDIndex idx = new NDIndex(String.valueOf(i) + "," +
                //        String.valueOf(j) + "," + String.valueOf(x.getInt(0, j)));
                if( x.getInt(0, j) == 0 )
                    sum += Px_y0.getDouble(i, j);  //[i][j][x[j]]
                if( x.getInt(0, j) == 1 )
                    sum += Px_y1.getDouble(i, j);
            }

            // 最后再和先验概率相加（也就是式4 .7 中的先验概率乘以后头那些东西，乘法因为log全变成了加法）
            P[i] = sum + Py.getDouble(0, i);
        }

        // max(P)：找到概率最大值
        // P.index(max(P))：找到该概率最大值对应的所有（索引值和标签值相等）
        return StatisticHelper.maxIndex(P) ;    //P.index(max(P));
    }

    /**
     * 通过训练集计算先验概率分布和条件概率分布
     * @param trainDataArr: 训练数据集
     * @param trainLabelArr: 训练标记集
     * @return: 先验概率分布和条件概率分布
     */
    public ArrayList<DataFrame> getAllProbability(DataFrame trainDataArr, DataFrame trainLabelArr) {

        ArrayList<DataFrame> Ps = new ArrayList<>();

        try (NDManager manager = NDManager.newBaseManager()) {
            // 设置样本特诊数目，数据集中手写图片为28*28，转换为向量是784维。
            //（我们的数据集已经从图像转换成784维的形式了，CSV格式内就是）
            int featureNum = 784;
            // 设置类别数目，0-9共十个类别
            int classNum = 10;

            // 初始化先验概率分布存放数组，后续计算得到的P(Y = 0)放在Py[0]中，以此类推
            // 数据长度为10行1列
            //NDArray Py = manager.zeros(new Shape(classNum, 1), DataType.FLOAT64); //np.zeros((classNum, 1))
            double [][] Py = new double[1][classNum];
            for(int k = 0; k < classNum; k++ )
                Py[0][k] = 0.0;

            // 对每个类别进行一次循环，分别计算它们的先验概率分布
            // 计算公式为书中"4.2节 朴素贝叶斯法的参数估计 公式4.8"
            int [] labmat = new int[trainLabelArr.nrow()];
            for( int i = 0; i < trainLabelArr.nrow(); i++ )
                labmat[i] = trainLabelArr.getInt(i, 0);

            //StatisticHelper.printVectorElements(labmat);

            NDArray nd = manager.create(labmat);
            // System.out.println(nd.getShape());

            for( int i = 0; i < classNum; i++ ) {
                // 下方式子拆开分析
                // np.mat(trainLabelArr) == i：将标签转换为矩阵形式，里面的每一位与i比较，若相等，该位变为Ture，反之False
                // np.sum(np.mat(trainLabelArr) == i):计算上一步得到的矩阵中Ture的个数，进行求和(直观上就是找所有label中有多少个
                // 为i的标记，求得4.8式P（Y = Ck）中的分子)
                // np.sum(np.mat(trainLabelArr) == i)) + 1：参考“4.2.3节 贝叶斯估计”，例如若数据集总不存在y=1的标记，也就是说
                // 手写数据集中没有1这张图，那么如果不加1，由于没有y=1，所以分子就会变成0，那么在最后求后验概率时这一项就变成了0，再
                // 和条件概率乘，结果同样为0，不允许存在这种情况，所以分子加1，分母加上K（K为标签可取的值数量，这里有10个数，取值为10）
                // 参考公式4.11
                // (len(trainLabelArr) + 10)：标签集的总长度+10.
                // ((np.sum(np.mat(trainLabelArr) == i)) + 1) / (len(trainLabelArr) + 10)：最后求得的先验概率

                // System.out.println(nd.eq( i ).sum());
                //Py.set( new NDIndex(String.valueOf(i)), (nd.eq(i).sum().toLongArray()[0] + 1)*1.0 /(classNum + trainLabelArr.nrow()));
                Py[0][i] = (nd.eq(i).sum().toLongArray()[0] + 1)*1.0 /(classNum + trainLabelArr.nrow());
                //System.out.println("Py[i]: " + Py[0][i]);
            }
            // 转换为log对数形式
            // log书中没有写到，但是实际中需要考虑到，原因是这样：
            // 最后求后验概率估计的时候，形式是各项的相乘（“4.1 朴素贝叶斯法的学习” 式4.7），这里存在两个问题：1.某一项为0时，结果为0.
            // 这个问题通过分子和分母加上一个相应的数可以排除，前面已经做好了处理。2.如果特诊特别多（例如在这里，需要连乘的项目有784个特征
            // 加一个先验概率分布一共795项相乘，所有数都是0-1之间，结果一定是一个很小的接近0的数。）理论上可以通过结果的大小值判断， 但在
            // 程序运行中很可能会向下溢出无法比较，因为值太小了。所以人为把值进行log处理。log在定义域内是一个递增函数，也就是说log（x）中，
            // x越大，log也就越大，单调性和原数据保持一致。所以加上log对结果没有影响。此外连乘项通过log以后，可以变成各项累加，简化了计算。
            // 在似然函数中通常会使用log的方式进行处理（至于此书中为什么没涉及，我也不知道）

            for(int k = 0; k < classNum; k++ )
                Py[0][k] = Math.log(Py[0][k]);
            //System.out.println(Py[0][0]);

            // 计算条件概率 Px_y=P（X=x|Y = y）
            // 计算条件概率分成了两个步骤，下方第一个大for循环用于累加，参考书中“4.2.3 贝叶斯估计 式4.10”，下方第一个大for循环内部是
            // 用于计算式4.10的分子，至于分子的+1以及分母的计算在下方第二个大For内
            // 初始化为全0矩阵，用于存放所有情况下的条件概率
            //NDArray Px_y = manager.zeros(new Shape(classNum, featureNum, 2), DataType.FLOAT64);
            double [][] Px_y0 = new double[classNum][featureNum];
            double [][] Px_y1 = new double[classNum][featureNum];
            for(int k = 0; k < classNum; k++ ) {
                for (int m = 0; m < featureNum; m++) {
                    Px_y0[k][m] = 0;
                    Px_y1[k][m] = 0;
                }
            }

            // 对标记集进行遍历
            for(int i = 0; i < trainLabelArr.nrow(); i++ ) {
                // 获取当前循环所使用的标记
                int label = trainLabelArr.getInt(i, 0);
                // 获取当前要处理的样本
                DataFrame x = trainDataArr.of(i);
                // 对该样本的每一维特诊进行遍历
                for( int j = 0; j < featureNum; j++ ) {
                    // 在矩阵中对应位置加1
                    // 这里还没有计算条件概率，先把所有数累加，全加完以后，在后续步骤中再求对应的条件概率
                    //NDIndex idx = new NDIndex(String.valueOf(label) + "," +
                    //                            String.valueOf(j) + "," + String.valueOf(x.getInt(0, j)));

                    if( x.getInt(0, j) == 0 ) Px_y0[label][j] += 1;
                    if( x.getInt(0, j) == 1 ) Px_y1[label][j] += 1;
                    //Px_y.set(idx, Px_y.get(idx).toDoubleArray()[0] + 1);
                }
            }

            // 第二个大for，计算式4.10的分母，以及分子和分母之间的除法
            // 循环每一个标记（共10个）
            for( int label = 0; label < classNum; label++ ) {
                // 循环每一个标记对应的每一个特征
                for( int j = 0; j < featureNum; j++ ) {
                    //NDIndex idx0 = new NDIndex(String.valueOf(label) + "," +
                    //        String.valueOf(j) + ",0");
                    //NDIndex idx1 = new NDIndex(String.valueOf(label) + "," +
                    //        String.valueOf(j) + ",1");

                    // 获取y=label，第j个特诊为0的个数
                    double Pxy0 = Px_y0[label][j];
                    // 获取y=label，第j个特诊为1的个数
                    double Pxy1 = Px_y1[label][j];

                    // 对式4.10的分子和分母进行相除，再除之前依据贝叶斯估计，分母需要加上2（为每个特征可取值个数）
                    // 分别计算对于y= label，x第j个特征为0和1的条件概率分布
                    Px_y0[label][j] = Math.log((Pxy0 + 1) / (Pxy0 + Pxy1 + 2));
                    Px_y1[label][j] = Math.log((Pxy1 + 1) / (Pxy0 + Pxy1 + 2));
                }
            }
            Ps.add(DataFrame.of(Py));
            Ps.add(DataFrame.of(Px_y0));
            Ps.add(DataFrame.of(Px_y1));
        }

        // 返回先验概率分布和条件概率分布
        return Ps;
    }

    /**
     * 测试准确率
     * @param Py: 先验概率分布
     * @param Px_y0: 条件概率分布
     * @param Px_y1: 条件概率分布
     * @param testDataList:待测试数据集
     * @param testLabelList: 待测试标签集
     * @return: 准确率
     */
    public double model_test(DataFrame Py, DataFrame Px_y0, DataFrame Px_y1,
                             DataFrame testDataList, DataFrame testLabelList) {
        // 错误次数计数
        int errorCnt = 0;
        int n_sample = testDataList.nrow();

        // 遍历测试集中每一个测试样本
        for( int i = 0; i < n_sample; i++ ) {
            // 获取预测值
            int predict = naiveBayes(Py, Px_y0, Px_y1, testDataList.of(i));
            // 与答案进行比较
            if( testLabelList.getInt(i, 0) != predict )
                errorCnt += 1;
        }

        // 返回准确率
        return 1.0 - (errorCnt*1.0 / n_sample);
    }

    public static void main(String [] args) {
        NaiveBayes NB = new NaiveBayes();
        String fnm = "./data/Mnist/mnist_train.csv";
        ArrayList<DataFrame> train_data = new ArrayList<>();
        ArrayList<DataFrame> test_data = new ArrayList<>();
        DataFrameHelper.loadMnistData(fnm, train_data);

        DataFrame trainDataArr = train_data.get(0);
        DataFrame trainLabelArr = train_data.get(1);

        // 开始训练，学习先验概率分布和条件概率分布
        System.out.println("start to train");
        ArrayList<DataFrame> Ps = NB.getAllProbability(trainDataArr, trainLabelArr);
        DataFrame Py   = Ps.get(0);
        DataFrame Px_y0 = Ps.get(1);
        DataFrame Px_y1 = Ps.get(2);
        System.out.println(Px_y0.getDouble(0, 0));
        System.out.println(Px_y1.getDouble(0, 0));

        // 获取测试集
        System.out.println("start read testSet");
        fnm = "./data/Mnist/mnist_test.csv";
        DataFrameHelper.loadMnistData(fnm, test_data);
        DataFrame testDataArr = test_data.get(0);
        DataFrame testLabelArr = test_data.get(1);

        // 使用习得的先验概率分布和条件概率分布对测试集进行测试
        System.out.println("start to test");
        double accuracy = NB.model_test(Py, Px_y0, Px_y1, testDataArr, testLabelArr);

        // 打印准确率
        System.out.println("The accuracy is: " + accuracy);

        System.exit(0);
    }
}
