package com.github.jiamny.ml.Ch10_LogisticRegressionAndMaximumEntropy;

import com.github.jiamny.ml.utils.DataFrameHelper;
import org.apache.commons.csv.CSVFormat;
import smile.data.DataFrame;
import smile.io.Read;

import java.util.ArrayList;
import java.util.HashMap;

import static com.github.jiamny.ml.utils.DataFrameHelper.range;
import static com.github.jiamny.ml.utils.StatisticHelper.maxIndex;

// 最大熵类
class MaxEnt {
    DataFrame trainDataList, trainLabelList, testDataList, testLabelList;
    int featureNum, N, n, M;
    ArrayList<HashMap<String, Integer>> fixy;
    ArrayList<HashMap<String, Integer>> xy2idDict;
    HashMap<Integer, String> id2xyDict;
    double [] Ep_xy, w;
    public MaxEnt(DataFrame trainDataList, DataFrame trainLabelList,
                       DataFrame testDataList, DataFrame testLabelList) {
        this.trainDataList = trainDataList;          //训练数据集
        this.trainLabelList = trainLabelList;        //训练标签集
        this.testDataList = testDataList;            //测试数据集
        this.testLabelList = testLabelList;          //测试标签集
        featureNum = trainDataList.ncol();           //特征数量

        N = trainDataList.nrow();                    //总训练集长度
        n = 0;                                       //训练集中（xi，y）对数量
        M = 10000;
        fixy = calc_fixy();                          //所有(x, y)对出现的次数
        w = new double[n];                           //Pw(y|x)中的w
        for( int i = 0; i < n; i++ )
            w[i] = 0.0;
        xy2idDict = new ArrayList<HashMap<String, Integer>>();
        createSearchDict();                          //(x, y)->id和id->(x, y)的搜索字典
        Ep_xy = calcEp_xy();                         //Ep_xy期望值
        System.out.println("n: " + n);
        System.out.println("N: " + N);
        System.out.println("Ep_xy[0]: " + Ep_xy[0]);
    }

    /**
     * 计算(x, y)在训练集中出现过的次数
     */
    public ArrayList<HashMap<String, Integer>> calc_fixy() {
        ArrayList<HashMap<String, Integer>> fixyDict = new ArrayList<HashMap<String, Integer>>();

        //建立特征数目个字典，属于不同特征的(x, y)对存入不同的字典中，保证不被混淆
         for( int i = 0; i < featureNum; i++ )
             fixyDict.add(i, new HashMap<String, Integer>());

        //遍历训练集中所有样本
        for( int i = 0; i < trainDataList.nrow(); i++ ) {
            // 遍历样本中所有特征
            for (int j = 0; j < featureNum; j++) {
                // 将出现过的(x, y)对放入字典中并计数值加1
                //fixyDict[j][(self.trainDataList[i][j], self.trainLabelList[i])] +=1;
                String s = "(" + trainDataList.getInt(i, j) + "," + trainLabelList.getInt(i, 0) + ")";
                if( fixyDict.get(j).containsKey(s) ) {
                    fixyDict.get(j).replace(s, fixyDict.get(j).get(s) + 1);
                } else {
                    fixyDict.get(j).put(s, 1);
                }
            }
        }
        // 对整个大字典进行计数，判断去重后还有多少(x, y)对，写入n
        for( HashMap<String, Integer> i : fixyDict )
            n += i.size();

        // 返回大字典
        return fixyDict;
    }

    /**
     * 计算特征函数f(x, y)关于经验分布P_(x, y)的期望值（下划线表示P上方的横线，
     * 同理Ep_xy中的“_”也表示p上方的横线）
     * 即“6.2.2 最大熵的定义”中第一个期望（82页最下方那个式子）
     * :return: 计算得到的Ep_xy
     */

    public double [] calcEp_xy() {
        // 初始化Ep_xy列表，长度为n
        double [] Ep_xy = new double[n];

        // 遍历每一个特征
        for( int feature = 0; feature < featureNum; feature++ ) {
            // 遍历每个特征中的(x, y)对
            // for (x, y) in self.fixy[feature] {
            HashMap<String, Integer> hash = fixy.get(feature);
            for( String key : hash.keySet() ) {
                // 获得其id
                int id = (xy2idDict.get(feature)).get(key);
                // 将计算得到的Ep_xy写入对应的位置中
                // fixy中存放所有对在训练集中出现过的次数，处于训练集总长度N就是概率了
                Ep_xy[id] = (fixy.get(feature)).get(key) *1.0 / N;
            }
        }

        // 返回期望
        return Ep_xy;
    }

    /**
     * 计算“6.23 最大熵模型的学习” 式6.22
     * @param X: 要计算的样本X（一个包含全部特征的样本）
     * @param y: 该样本的标签
     * @return: 计算得到的Pw(Y|X)
     */
    public double calcPwy_x(DataFrame X, int y) {
        //分子
        double numerator = 0.0;
        //分母
        double Z = 0.0;
        // 对每个特征进行遍历
        for(int i : range(featureNum) ) {
            // 如果该(xi,y)对在训练集中出现过
            String key = "(" + X.getInt(0, i) + "," + y + ")";
            if(xy2idDict.get(i).containsKey(key)) { //(X[i], y) in self.xy2idDict[i]
                // 在xy->id字典中指定当前特征i，以及(x, y)对：(X[i], y)，读取其id
                int index = xy2idDict.get(i).get(key); //[(X[i], y)]
                // 分子是wi和fi(x，y)的连乘再求和，最后指数
                // 由于当(x, y)存在时fi(x，y)为1，因为xy对肯定存在，所以直接就是1
                // 对于分子来说，就是n个wi累加，最后再指数就可以了
                // 因为有n个w，所以通过id将w与xy绑定，前文的两个搜索字典中的id就是用在这里
                numerator += w[index];
                //System.out.println("numerator: " + numerator);
            }
            // 同时计算其他一种标签y时候的分子，下面的z并不是全部的分母，再加上上式的分子以后
            // 才是完整的分母，即z = z + numerator
            key = "(" + X.getInt(0, i) + "," + (1 - y) + ")";
            if(xy2idDict.get(i).containsKey(key)) {     // (X[i], 1-y) in self.xy2idDict[i]:
                // 原理与上式相同
                int index = xy2idDict.get(i).get(key);  //xy2idDict[i][(X[i], 1 - y)]
                Z += w[index];
            }
        }
        //System.out.println("numerator: " + numerator + " Z: " + Z);
        // 计算分子的指数
        numerator = Math.exp(numerator);
        // 计算分母的z
        Z = Math.exp(Z) + numerator;
        //System.out.println("after exp numerator: " + numerator + " Z: " + Z);
        // 返回Pw(y|x)
        return numerator / Z;
    }

    /**
     * 计算特征函数f(x, y)关于模型P(Y|X)与经验分布P_(X, Y)的期望值（P后带下划线“_”表示P上方的横线
     * 程序中部分下划线表示“|”，部分表示上方横线，请根据具体公式自行判断,）
     * 即“6.2.2 最大熵模型的定义”中第二个期望（83页最上方的期望）
     * @return
     */
    public double [] calcEpxy() {
        // 初始化期望存放列表，对于每一个(x, y)对都有一个期望
        // 这里的x是单个的特征，不是一个样本的全部特征。例如x={x1，x2，x3.....，xk}，实际上是（x1，y），（x2，y），。。。
        // 但是在存放过程中需要将不同特诊的分开存放，李航的书可能是为了公式的泛化性高一点，所以没有对这部分提及
        // 具体可以看我的博客，里面有详细介绍  www.pkudodo.com
        double [] Epxy = new double[n];

        //对于每一个样本进行遍历
        for(int i : range(N) ) {
            //初始化公式中的P(y|x)列表
            double[] Pwxy = new double[2];
            // 计算P(y = 0 } X)
            // 注：程序中X表示是一个样本的全部特征，x表示单个特征，这里是全部特征的一个样本
            Pwxy[0] = calcPwy_x(trainDataList.of(i), 0);
            // 计算P(y = 1 } X)
            Pwxy[1] = calcPwy_x(trainDataList.of(i), 1);

            for (int feature : range(featureNum)) {
                for (int y : range(2)) {
                    String key = "(" + trainDataList.getInt(i, feature) + "," + y + ")";
                    if (fixy.get(feature).containsKey(key)) {
                        int id = xy2idDict.get(feature).get(key);    //[feature][(self.trainDataList[i][feature], y)]
                        Epxy[id] += (1.0 / N) * Pwxy[y];
                    }
                }
            }
        }
        return Epxy;
    }

    public void maxEntropyTrain(int iter) {
        if( iter < 100 )
            iter = 100;

        //  设置迭代次数寻找最优解
        for(int i = 0; i < iter; i++ ) {
            // 单次迭代起始时间点
            //    iterStart = time.time()
            Long iterStart = System.currentTimeMillis();

            // 计算“6.2 .3 最大熵模型的学习”中的第二个期望（83 页最上方哪个）
            double [] Epxy = calcEpxy();

            // 使用的是IIS，所以设置sigma列表
            double [] sigmaList = new double[n];
            for (int j = 0; j < n; j++)
                sigmaList[j] = 0.0;

            // 对于所有的n进行一次遍历
            for (int j = 0; j < n; j++)
                // 依据“6.3 .1 改进的迭代尺度法”式6 .34 计算
                sigmaList[j] = (1.0 / M) * Math.log(Ep_xy[j] / Epxy[j]);

            // 按照算法6 .1 步骤二中的（b）更新w
            //w = [self.w[i] + sigmaList[i] for i in range(self.n)]
            for(int k : range(n) )
                w[k] = w[k] + sigmaList[k];

            // 单次迭代结束
            Long iterEnd = System.currentTimeMillis();

            // 打印运行时长信息
            System.out.printf("iter: %d/%d, time: %ds\n" , i, iter, (iterEnd - iterStart)/1000 );
        }
    }

    /**
     * 创建查询字典
     *  xy2idDict：通过(x,y)对找到其id,所有出现过的xy对都有一个id
     *  id2xyDict：通过id找到对应的(x,y)对
     */
    public void createSearchDict() {
        // 设置xy搜多id字典
        // 这里的x指的是单个的特征，而不是某个样本，因此将特征存入字典时也需要存入这是第几个特征
        // 这一信息，这是为了后续的方便，否则会乱套。
        // 比如说一个样本X = (0, 1, 1) label =(1)
        // 生成的标签对有(0, 1), (1, 1), (1, 1)，三个(x，y)对并不能判断属于哪个特征的，后续就没法往下写
        // 不可能通过(1, 1)就能找到对应的id，因为对于(1, 1),字典中有多重映射
        // 所以在生成字典的时总共生成了特征数个字典，例如在mnist中样本有784维特征，所以生成784个字典，属于
        // 不同特征的xy存入不同特征内的字典中，使其不会混淆
        for( int i : range(featureNum) ) {
            xy2idDict.add(i, new HashMap<String, Integer>());
        }
        // 初始化id到xy对的字典。因为id与(x，y)的指向是唯一的，所以可以使用一个字典
        id2xyDict = new HashMap<Integer, String>();

        // 设置缩影，其实就是最后的id
        int index = 0;
        // 对特征进行遍历
        for(int feature : range(featureNum) ) {
            // 对出现过的每一个(x, y)对进行遍历
            // fixy：内部存放特征数目个字典，对于遍历的每一个特征，单独读取对应字典内的(x, y)对
            for (String key : fixy.get(feature).keySet()) {
                // 将该(x, y)对存入字典中，要注意存入时通过[feature]指定了存入哪个特征内部的字典
                // 同时将index作为该对的id号
                xy2idDict.get(feature).put(key, index);
                // 同时在id->xy字典中写入id号，val为(x, y)对
                id2xyDict.put(index, key);
                //id加一
                index += 1;
            }
        }
    }

    /**
     * 预测标签
     * @param X:要预测的样本
     * @return: 预测值
     */
   public int predict(DataFrame X) {
       // 因为y只有0和1，所有建立两个长度的概率列表
       double [] result = new double[2];

       // 循环计算两个概率
       for(int i : range(2) ) {
           // 计算样本x的标签为i的概率
           result[i] = calcPwy_x(X, i);
       }
       //System.out.println("result[0]: " + result[0] + " result[1]: " + result[1]);
       // 返回标签
       // max(result)：找到result中最大的那个概率值
       // result.index(max(result))：通过最大的那个概率值再找到其索引，索引是0就返回0，1就返回1
       return maxIndex(result);       //result.index(max(result))
   }

    /**
     * 对测试集进行测试
     * @return:
     */
    public double test() {
        // 错误值计数
        int errorCnt = 0;
        // 对测试集中所有样本进行遍历
        for( int i : range(testDataList.nrow())) {
            // 预测该样本对应的标签
            int result = predict(testDataList.of(i));
            // 如果错误，计数值加1
            if( result != testLabelList.getInt(i, 0) )
                errorCnt += 1;
        }
        // 返回准确率
        return (1.0 - errorCnt*1.0 / testDataList.nrow());
    }
}

public class MaxEntropy {

    public void loadData(String fileName, ArrayList<DataFrame> tdt) {
        try {
            var format = CSVFormat.newFormat(',');
            DataFrame mnist_train = Read.csv(fileName, format);

            int [] label_idx = new int[1];
            label_idx[0] = 0;
            DataFrame train_labels = mnist_train.select(label_idx);

            // Mnsit有0-9是个标记，由于是二分类任务，所以将标记0的作为1，其余为0
            int [][] L = new int[train_labels.nrow()][1];
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

            //在放入的同时将原先字符串形式的数据转换为整型
            //此外将数据进行了二值化处理，大于128的转换成1，小于的转换成0，方便后续计算
            int [][] mdt = new int[train_data.nrow()][train_data.ncol()];

            for(int r = 0; r < train_data.nrow(); r++) {
                for(int c = 0; c < train_data.ncol(); c++) {
                    if(train_data.getDouble(r, c) > 128) {
                        mdt[r][c] = 1;
                    } else {
                        mdt[r][c] = 0;
                    }
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

        MaxEntropy ME = new MaxEntropy();
        String fnm = "./data/Mnist/mnist_train.csv";
        ArrayList<DataFrame> train_data = new ArrayList<>();
        ME.loadData(fnm, train_data);

        fnm = "./data/Mnist/mnist_test.csv";
        ArrayList<DataFrame> test_data = new ArrayList<>();
        ME.loadData(fnm, test_data);

        // 初始化最大熵类
        MaxEnt maxEnt = new MaxEnt(train_data.get(0), train_data.get(1), test_data.get(0), test_data.get(1));

        // 开始训练
        System.out.println("start to train");
        maxEnt.maxEntropyTrain(10);

        // 开始测试
        System.out.println("start to test");
        double accuracy = maxEnt.test();
        System.out.println("the accuracy is: " + accuracy);

        // 打印时间
        System.out.printf("time span: %ds\n", (System.currentTimeMillis() - start)/1000);

        System.exit(0);
    }
}
