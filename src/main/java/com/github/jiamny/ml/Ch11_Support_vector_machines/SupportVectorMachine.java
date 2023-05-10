package com.github.jiamny.ml.Ch11_Support_vector_machines;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import com.github.jiamny.ml.utils.DataFrameHelper;
import com.github.jiamny.ml.utils.StatisticHelper;
import org.apache.commons.csv.CSVFormat;
import smile.data.DataFrame;
import smile.io.Read;
import smile.math.matrix.Matrix;
import smile.math.random.UniversalGenerator;

import java.util.ArrayList;
import java.util.Random;

import static com.github.jiamny.ml.utils.DataFrameHelper.range;
import static com.github.jiamny.ml.utils.StatisticHelper.printVectorElements;

//SVM类
class SVM {

    Matrix trainDataMat, trainLabelMat;
    int m, n;
    double sigma, C, toler, b;
    double [] alpha, E;
    double [][] k;
    ArrayList<Integer> supportVecIndex;
    /**
     * SVM相关参数初始化
     * @param trainDataList:训练数据集
     * @param trainLabelList: 训练测试集
     * @param sigma: 高斯核中分母的σ
     * @param C:软间隔中的惩罚参数
     * @param toler:松弛变量
     */
    //sigma =10, double C =200, double toler =0.001
    public SVM( DataFrame trainDataList, DataFrame trainLabelList,
               double sigma, double C, double toler ) {
        // 注：
        // 关于这些参数的初始值：参数的初始值大部分没有强要求，请参照书中给的参考，例如C是调和间隔与误分类点的系数，
        // 在选值时通过经验法依据结果来动态调整。（本程序中的初始值参考于《机器学习实战》中SVM章节，因为书中也
        // 使用了该数据集，只不过抽取了很少的数据测试。参数在一定程度上有参考性。）
        // 如果使用的是其他数据集且结果不太好，强烈建议重新通读所有参数所在的公式进行修改。例如在核函数中σ的值
        // 高度依赖样本特征值范围，特征值范围较大时若不相应增大σ会导致所有计算得到的核函数均为0

        trainDataMat = trainDataList.toMatrix();        // 训练数据集
        trainLabelMat = trainLabelList.toMatrix();      // 训练标签集，为了方便后续运算提前做了转置，变为列向量

        m = trainDataMat.nrow(); n = trainDataMat.ncol(); // m：训练集数量 n：样本特征数目
        this.sigma = sigma;                               // 高斯核分母中的σ
        this.C = C;                                       // 惩罚参数
        this.toler = toler;                               // 松弛变量

        k = calcKernel();                                 // 核函数（初始化时提前计算）
        b = 0.0;                                          // SVM中的偏置b
        alpha = new double[trainDataMat.nrow()];          // α 长度为训练集数目
        for(int i : range(trainDataMat.nrow()))
            alpha[i] = 0;

        E = new double[trainLabelMat.nrow()];             // [0*self.trainLabelMat[i,0] for i in range(self.trainLabelMat.shape[0])]     //SMO运算过程中的Ei
        for(int i : range(trainDataMat.nrow()))
            E[i] = 0;
        supportVecIndex = new ArrayList<Integer>();
    }

    /**
     * 计算核函数
     * 使用的是高斯核 详见“7.3.3 常用核函数” 式7.90
     * @return: 高斯核矩阵
     */
    public double [][] calcKernel() {
        // 初始化高斯核结果矩阵 大小 = 训练集长度m * 训练集长度m
        // k[i][j] = Xi * Xj
        k = new double[m][m];   //[[0 for i in range(self.m)] for j in range(self.m)]
        for(int i : range(m) ) {
            for(int j : range(m) )
                k[i][j] = 0;
        }

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray data = manager.create(trainDataMat.toArray());
            data.setRequiresGradient(false);

            // 大循环遍历Xi，Xi为式7.90中的x
            for(int  i : range(m) ) {
                // 每100个打印一次
                // 不能每次都打印，会极大拖慢程序运行速度
                // 因为print是比较慢的
                if( i % 100 == 0 )
                    System.out.printf("construct the kernel: %d/%d\n", i, m);
                // 得到式7 .90 中的X
                NDArray X = data.get(i);
                // 小循环遍历Xj，Xj为式7 .90 中的Z
                // 由于 Xi *Xj 等于 Xj * Xi，一次计算得到的结果可以
                // 同时放在k[i][j] 和k[j][i] 中，这样一个矩阵只需要计算一半即可
                // 所以小循环直接从i开始
                for (int j : range(m)) {
                    // 获得Z
                    NDArray Z = data.get(j);
                    // 先计算 || X - Z ||^2
                    NDArray T = X.sub(Z);
                    double result = T.matMul(T.transpose()).getDouble();

                    // 分子除以分母后去指数，得到的即为高斯核结果
                    // np.exp(-1 * result / (2 * self.sigma * * 2))
                    result = Math.exp((-1*result)/(2*Math.pow(sigma, 2)));
                    //System.out.println("result: " + result);
                    // 将Xi * Xj的结果存放入k[i][j] 和k[j][i] 中
                    k[i][j] = result;
                    k[j][i] = result;
                }
            }
        }

        // 返回高斯核矩阵
        return k;
    }

    /**
     * 查看第i个α是否满足KKT条件
     * @param i:α的下标
     * :return:
     *     True：满足
     *     False：不满足
     */
    public boolean isSatisfyKKT(int i) {
        double gxi = calc_gxi(i);
        double yi = trainLabelMat.get(i, 0); //[i]

        // 判断依据参照“7.4.2 变量的选择方法”中“1.第1个变量的选择”
        // 式7.111到7.113
        // --------------------
        // 依据7.111

        if((Math.abs(alpha[i]) < toler) && (yi * gxi >= 1))
            return true;
        // 依据7.113
        else if ((Math.abs(alpha[i] - C) < toler) && (yi * gxi <= 1))
            return true;
        // 依据7.112
        else if( (alpha[i] > - toler) && (alpha[i] < (C + toler)) && (Math.abs(yi * gxi - 1) < toler))
            return true;

        return false;
    }

    /**
     * 计算g(xi)
     * 依据“7.101 两个变量二次规划的求解方法”式7.104
     * @param i:x的下标
     * @return: g(xi)的值
     */
    public double calc_gxi(int i) {
        // 初始化g(xi)
        double gxi = 0;
        // 因为g(xi)是一个求和式+b的形式，普通做法应该是直接求出求和式中的每一项再相加即可
        // 但是读者应该有发现，在“7.2.3 支持向量”开头第一句话有说到“对应于α>0的样本点
        // (xi, yi)的实例xi称为支持向量”。也就是说只有支持向量的α是大于0的，在求和式内的
        // 对应的αi*yi*K(xi, xj)不为0，非支持向量的αi*yi*K(xi, xj)必为0，也就不需要参与
        // 到计算中。也就是说，在g(xi)内部求和式的运算中，只需要计算α>0的部分，其余部分可
        // 忽略。因为支持向量的数量是比较少的，这样可以再很大程度上节约时间
        // 从另一角度看，抛掉支持向量的概念，如果α为0，αi*yi*K(xi, xj)本身也必为0，从数学
        // 角度上将也可以扔掉不算
         // index获得非零α的下标，并做成列表形式方便后续遍历
        ArrayList<Integer> index = new ArrayList<>();
        //index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        for( int k = 0; k < alpha.length; k++ )
            if( alpha[k] != 0 )
                index.add(k);

        // 遍历每一个非零α，i为非零α的下标
        for(int j : index.stream().mapToInt(j -> j.intValue()).toArray())
            // 计算g(xi)
            gxi += alpha[j] * trainLabelMat.get(j, 0) * k[j][i];
        // 求和结束后再单独加上偏置b
        gxi += b;
        //System.out.println("gxi: " + gxi);
        // 返回
        return gxi;
    }

    /**
     * 计算Ei
     * 根据“7.4.1 两个变量二次规划的求解方法”式7.105
     * @param i: E的下标
     * @return:
     */
    public double calcEi(int i) {
        // 计算g(xi)
        double gxi = calc_gxi(i);
        //Ei = g(xi) - yi,直接将结果作为Ei返回
        return (gxi - trainLabelMat.get(i, 0));
    }

    /**
     * SMO中选择第二个变量
     * @param E1: 第一个变量的E1
     * @param i: 第一个变量α的下标
     * @return: E2，α2的下标
     */
    public ArrayList<String> getAlphaJ(double E1, int i) {
        //初始化E2
        double E2 = 0;
        // 初始化|E1-E2|为-1
        double maxE1_E2 = -1;
        // 初始化第二个变量的下标
        int maxIndex = -1;

        // 这一步是一个优化性的算法
        // 实际上书上算法中初始时每一个Ei应当都为-yi（因为g(xi)由于初始α为0，必然为0）
        // 然后每次按照书中第二步去计算不同的E2来使得|E1-E2|最大，但是时间耗费太长了
        // 作者最初是全部按照书中缩写，但是本函数在需要3秒左右，所以进行了一些优化措施
        // --------------------------------------------------
        // 在Ei的初始化中，由于所有α为0，所以一开始是设置Ei初始值为-yi。这里修改为与α
        // 一致，初始状态所有Ei为0，在运行过程中再逐步更新
        // 因此在挑选第二个变量时，只考虑更新过Ei的变量，但是存在问题
        // 1.当程序刚开始运行时，所有Ei都是0，那挑谁呢？
        //   当程序检测到并没有Ei为非0时，将会使用随机函数随机挑选一个
        // 2.怎么保证能和书中的方法保持一样的有效性呢？
        //   在挑选第一个变量时是有一个大循环的，它能保证遍历到每一个xi，并更新xi的值，
        // 在程序运行后期后其实绝大部分Ei都已经更新完毕了。下方优化算法只不过是在程序运行
        // 的前半程进行了时间的加速，在程序后期其实与未优化的情况无异
        // ------------------------------------------------------

        // 获得Ei非0的对应索引组成的列表，列表内容为非0Ei的下标i
        // nozeroE = [i for i, Ei in enumerate(self.E) if Ei != 0]
        ArrayList<Integer> nozeroE = new ArrayList<>();
        for (int k = 0; k < E.length; k++) {
            if( E[k] != 0 )
                nozeroE.add(k);
        }

        if( ! nozeroE.isEmpty() ) {
            // 对每个非零Ei的下标i进行遍历
            for (int j : nozeroE.stream().mapToInt(j -> j.intValue()).toArray()) {
                // 计算E2
                double E2_tmp = calcEi(j);
                // 如果|E1-E2|大于目前最大值
                if (Math.abs(E1 - E2_tmp) > maxE1_E2) {
                    // 更新最大值
                    maxE1_E2 = Math.abs(E1 - E2_tmp);

                    // 更新最大值E2的索引j
                    maxIndex = j;
                    // 更新最大值E2
                    E2 = E2_tmp;
                }
            }
        }

        // 如果列表中没有非0元素了（对应程序最开始运行时的情况）
        if( maxIndex == -1 ) {
            maxIndex = i;

            UniversalGenerator rand = new UniversalGenerator();
            while (maxIndex == i) {
                // 获得随机数，如果随机数与第一个变量的下标i一致则重新随机
                maxIndex = rand.nextInt(m);       //int(random.uniform(0, m));
            }

            // 获得E2
            E2 = calcEi(maxIndex);
        }

        // 返回第二个变量的E2值以及其索引
        ArrayList<String> rlt = new ArrayList<>();
        rlt.add(String.valueOf(E2));
        rlt.add(String.valueOf(maxIndex));
        return rlt;
    }

    public void train(int iter) {
        // iterStep：迭代次数，超过设置次数还未收敛则强制停止
        // parameterChanged：单次迭代中有参数改变则增加1
        int iterStep = 0;
        int parameterChanged = 1;

        // 如果没有达到限制的迭代次数以及上次迭代中有参数改变则继续迭代
        // parameterChanged==0时表示上次迭代没有参数改变，如果遍历了一遍都没有参数改变，说明
        // 达到了收敛状态，可以停止了
        while( (iterStep < iter) && (parameterChanged > 0) ) {
            // 打印当前迭代轮数
            System.out.printf("iter:%d:%d  \n", iterStep, iter);
            // 迭代步数加1
            iterStep += 1;
            // 新的一轮将参数改变标志位重新置0
            parameterChanged = 0;

            //大循环遍历所有样本，用于找SMO中第一个变量
            for( int i : range(m) ) {
                // 查看第一个遍历是否满足KKT条件，如果不满足则作为SMO中第一个变量从而进行优化
                if( ! isSatisfyKKT(i) ) {
                    // 如果下标为i的α不满足KKT条件，则进行优化
                    // 第一个变量α的下标i已经确定，接下来按照“7.4.2 变量的选择方法”第二步
                    // 选择变量2。由于变量2的选择中涉及到|E1 - E2|，因此先计算E1
                    double E1 = calcEi(i);

                    // 选择第2个变量
                    ArrayList<String> rlt = getAlphaJ(E1, i);
                    double E2 = Double.parseDouble( rlt.get(0) );
                    int j = Integer.parseInt( rlt.get(1) );

                    // 参考“7.4.1两个变量二次规划的求解方法” P126 下半部分
                    // 获得两个变量的标签
                    double y1 = trainLabelMat.get(i, 0);
                    double y2 = trainLabelMat.get(j, 0);

                    // 复制α值作为old值
                    double alphaOld_1 = alpha[i];
                    double alphaOld_2 = alpha[j];

                    // 依据标签是否一致来生成不同的L和H
                    double L, H;
                    if( y1 != y2 ) {
                        L = Math.max(0, alphaOld_2 - alphaOld_1);
                        H = Math.min(C, C + alphaOld_2 - alphaOld_1);
                    } else {
                        L = Math.max(0, alphaOld_2 + alphaOld_1 - C);
                        H = Math.min(C, alphaOld_2 + alphaOld_1);
                    }

                    // 如果两者相等，说明该变量无法再优化，直接跳到下一次循环
                    if( L == H )
                        continue;

                    // 计算α的新值
                    // 依据“7.4.1两个变量二次规划的求解方法”式7.106更新α2值
                    // 先获得几个k值，用来计算事7.106中的分母η
                    double k11 = k[i][i];
                    double k22 = k[j][j];
                    double k21 = k[j][i];
                    double k12 = k[i][j];

                    // 依据式7.106更新α2，该α2还未经剪切
                    double alphaNew_2 = alphaOld_2 + y2 * (E1 - E2) / (k11 + k22 - 2 * k12);

                    // 剪切α2
                    if( alphaNew_2 < L)
                        alphaNew_2 = L;
                    else if( alphaNew_2 > H )
                        alphaNew_2 = H;

                    // 更新α1，依据式7.109
                    double alphaNew_1 = alphaOld_1 + y1 * y2 * (alphaOld_2 - alphaNew_2);

                    // 依据“7.4.2 变量的选择方法”第三步式7.115和7.116计算b1和b2
                    double b1New = -1 * E1 - y1 * k11 * (alphaNew_1 - alphaOld_1)
                    - y2 * k21 * (alphaNew_2 - alphaOld_2) + b;

                    double b2New = -1 * E2 - y1 * k12 * (alphaNew_1 - alphaOld_1)
                    - y2 * k22 * (alphaNew_2 - alphaOld_2) + b;

                    // 依据α1和α2的值范围确定新b
                    double bNew;
                    if( (alphaNew_1 > 0) && (alphaNew_1 < C) ) {
                        bNew = b1New;
                    } else if( (alphaNew_2 > 0) && (alphaNew_2 < C) )
                        bNew = b2New;
                    else
                        bNew = (b1New + b2New) / 2;

                    // 将更新后的各类值写入，进行更新
                    alpha[i] = alphaNew_1;
                    alpha[j] = alphaNew_2;
                    b = bNew;

                    E[i] = calcEi(i);
                    E[j] = calcEi(j);

                    // 如果α2的改变量过于小，就认为该参数未改变，不增加parameterChanged值
                    // 反之则自增1

                    if( Math.abs(alphaNew_2 - alphaOld_2) >= 0.00001 )
                        parameterChanged += 1;

                    // 打印迭代轮数，i值，该迭代轮数修改α数目
                    System.out.printf("iter: %d i:%d, pairs changed %d\n", iterStep, i, parameterChanged);
                }
            }
        }
        // 全部计算结束后，重新遍历一遍α，查找里面的支持向量
        for( int i : range(m) )
            // 如果α>0，说明是支持向量
            if( alpha[i] > 0 )
                // 将支持向量的索引保存起来
                supportVecIndex.add(i);
    }

    /**
     * 单独计算核函数
     * @param x1:向量1
     * @param x2: 向量2
     * @return: 核函数结果
     */
    public double calcSinglKernel(Matrix x1, Matrix x2) {

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray d1 = manager.create(x1.toArray()).stopGradient();
            NDArray d2 = manager.create(x2.toArray()).stopGradient();

            // 按照“7.3.3 常用核函数”式7.90计算高斯核
            NDArray T = d1.sub(d2);
            double result = T.matMul(T.transpose()).getDouble();
            result = Math.exp((-1 * result) / (2 * Math.pow(sigma, 2)));

            // 返回结果
            return Math.exp(result);
        }
    }

    /**
     * 对样本的标签进行预测
     * 公式依据“7.3.4 非线性支持向量分类机”中的式7.94
     * @param x: 要预测的样本x
     * @return: 预测结果
     */
    public int predict(Matrix x) {
        int result = 0;
        for( int i : supportVecIndex.stream().mapToInt(i -> i.intValue()).toArray()) {
            // 遍历所有支持向量，计算求和式
            // 如果是非支持向量，求和子式必为0，没有必须进行计算
            // 这也是为什么在SVM最后只有支持向量起作用
            // ----------------------------------------------------------------
            // 先单独将核函数计算出来
            double tmp = calcSinglKernel(trainDataMat.submatrix(i, 0, i, n-1).clone(), x);
            // 对每一项子式进行求和，最终计算得到求和项的值
            result += alpha[i] * trainLabelMat.get(i, 0) * tmp;
        }
        // 求和项计算结束后加上偏置b
        result += b;
        // 使用sign函数返回预测结果
        return StatisticHelper.sign(result);
    }

    /**
     * 测试
     * @param testDataList:测试数据集
     * @param testLabelList: 测试标签集
     * @return: 正确率
     */
    public double test(DataFrame testDataList, DataFrame testLabelList) {
        Matrix testDataMat = testDataList.toMatrix();
        Matrix testLabelMat = testLabelList.toMatrix();

        // 错误计数值
        int errorCnt = 0;
        // 遍历测试集所有样本
        for( int i : range(testDataMat.nrow()) ) {
            // 打印目前进度
            System.out.printf("test:%d:%d\n", i, testDataMat.nrow());

            // 获取预测结果
            int result = predict( testDataMat.submatrix(i, 0, i, n - 1) );

            // 如果预测与标签不一致，错误计数值加一
            if (result != testLabelMat.get(i, 0))
                errorCnt += 1;
        }
        // 返回正确率
        return( (1.0 - (errorCnt*1.0 / testDataMat.nrow()))*100);
    }
}

public class SupportVectorMachine {

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
            int [][] L = new int[train_labels.nrow()][1]; //train_labels.toMatrix().toArray();
            for( int r = 0; r < train_labels.nrow(); r++ )
                if( train_labels.get(r, 0) == Integer.valueOf(0) )
                    L[r][0] = 1;
                else
                    L[r][0] = -1;
            train_labels = DataFrame.of(L);


            int [] data_idx = new int[mnist_train.ncol()-1];
            for(int i = 1; i <= (mnist_train.ncol()-1); i++ )
                data_idx[i - 1] = i;
            DataFrame train_data = mnist_train.select(data_idx);

            // 将所有数据除255归一化
            double [][] mdt = new double[train_data.nrow()][train_data.ncol()];

            for(int r = 0; r < train_data.nrow(); r++) {
                for(int c = 0; c < train_data.ncol(); c++) {
                    mdt[r][c] = ((train_data.getInt(r, c)*1.0)/255.0);
                }
            }
            DataFrame ntrain_data = DataFrame.of(mdt);

            tdt.add(ntrain_data);
            tdt.add(train_labels);
        } catch(Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String [] args) {
        Long start = System.currentTimeMillis();

        SupportVectorMachine SVM = new SupportVectorMachine();
        String fnm = "./data/Mnist/mnist_train.csv";
        ArrayList<DataFrame> train_data = new ArrayList<>();
        SVM.loadData(fnm, train_data);

        //初始化SVM类
        System.out.println("start init SVM");
        SVM svm = new SVM(train_data.get(0).slice(0, 1000),
                train_data.get(1).slice(0, 1000), 10, 200, 0.001);
        svm.train(100);

        fnm = "./data/Mnist/mnist_test.csv";
        ArrayList<DataFrame> test_data = new ArrayList<>();
        SVM.loadData(fnm, test_data);

        System.out.println("The accur is: " + svm.test( test_data.get(0).slice(0, 100),
                test_data.get(1).slice(0, 100)));

        System.out.printf("time span: %ds\n", (System.currentTimeMillis() - start)/1000);

        System.exit(0);
    }
}
