package com.github.jiamny.ml.Ch12_Boosting;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.apache.commons.csv.CSVFormat;
import smile.data.DataFrame;
import smile.io.Read;
import smile.math.matrix.Matrix;

import java.util.ArrayList;
import java.util.Arrays;

import static com.github.jiamny.ml.utils.DataFrameHelper.range;
import static com.github.jiamny.ml.utils.StatisticHelper.sign;

class SingleBoostTree {
    public static final long serialVersionUID = 438L;
    public double e;
    public double div;
    public String rule;
    public double alpha;
    public int feature;
    public ArrayList<Integer> Gx;
    public SingleBoostTree() {
        e = 0; div = 0; alpha = 0;
        feature = -1;
        rule = "";
        Gx = null;
    }
}
public class AdaBoost {
    
    public void loadData(String fileName, ArrayList<DataFrame> tdt) {
        try {
            var format = CSVFormat.newFormat(',');
            DataFrame mnist_train = Read.csv(fileName, format);

            // 将标记信息放入标记集中
            int [] label_idx = new int[1];
            label_idx[0] = 0;
            DataFrame train_labels = mnist_train.select(label_idx);

            // 转换成二分类任务
            // 标签0设置为1，反之为-1
            int [][] L = new int[train_labels.nrow()][1];
            for( int r = 0; r < train_labels.nrow(); r++ )
                if( train_labels.get(r, 0) == Integer.valueOf(0) )
                    L[r][0] = 1;
                else
                    L[r][0] = -1;
            train_labels = DataFrame.of(L);

            // 此外将数据进行了二值化处理，大于128的转换成1，小于的转换成0，方便后续计算
            int [] data_idx = new int[mnist_train.ncol()-1];
            for(int i = 1; i <= (mnist_train.ncol()-1); i++ )
                data_idx[i - 1] = i;
            DataFrame train_data = mnist_train.select(data_idx);

            int [][] mdt = new int[train_data.nrow()][train_data.ncol()];

            for(int r = 0; r < train_data.nrow(); r++) {
                for(int c = 0; c < train_data.ncol(); c++) {
                    if( train_data.getInt(r, c) > 128 )
                        mdt[r][c] = 1;
                    else
                        mdt[r][c] = 0;
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
     * 计算分类错误率
     * @param trainDataArr:训练数据集数字
     * @param trainLabelArr: 训练标签集数组
     * @param n: 要操作的特征
     * @param div:划分点
     * @param rule:正反例标签
     * @param D:权值分布D
     * @return:预测结果， 分类误差率
     */
    public static double Tree_E = 0.0;
    public ArrayList<Integer> calc_e_Gx(DataFrame trainDataArr, DataFrame trainLabelArr,
                            int n, double div, String rule, NDArray D) {
        // 初始化分类误差率为0
        Tree_E = 0.0;
        // 将训练数据矩阵中特征为n的那一列单独剥出来做成数组。因为其他元素我们并不需要，
        // 直接对庞大的训练集进行操作的话会很慢
        Matrix x = trainDataArr.select(n).toMatrix();// [:,n]
        // 同样将标签也转换成数组格式，x和y的转换只是单纯为了提高运行速度
        // 测试过相对直接操作而言性能提升很大
        DataFrame y = trainLabelArr;
        //predict = []
        ArrayList<Integer> predict = new ArrayList<Integer>() ;

        int L = 0, H = 0;
        // 依据小于和大于的标签依据实际情况会不同，在这里直接进行设置
        if( rule.equals("LisOne") ) {
            L = 1;
            H = -1;
        } else {
            L = -1;
            H = 1;
        }
        //printVectorElements(D.toDoubleArray());

        // 遍历所有样本的特征m
        for(int i : range(trainDataArr.nrow()) ) {
            if (x.get(i, 0) < div) {
                // 如果小于划分点，则预测为L
                // 如果设置小于div为1，那么L就是1，
                // 如果设置小于div为-1，L就是-1
                predict.add(L);
                // 如果预测错误，分类错误率要加上该分错的样本的权值（8.1式）
                if (y.getInt(i, 0) != L)
                    Tree_E += D.toDoubleArray()[i];
            } else { //if (x.get(i, 0) >= div) {
                // 与上面思想一样
                predict.add(H);
                if (y.getInt(i, 0) != H)
                    Tree_E += D.toDoubleArray()[i];
            }
        }

        // 返回预测结果和分类错误率e
        // 预测结果其实是为了后面做准备的，在算法8.1第四步式8.4中exp内部有个Gx，要用在那个地方
        // 以此来更新新的D
        // np.array(predict), e
        return predict;
    }

    /**
     * 创建单层提升树
     * @param trainDataArr:训练数据集数组
     * @param trainLabelArr: 训练标签集数组
     * @param D: 算法8.1中的D
     * @return: 创建的单层提升树
     */
    SingleBoostTree createSingleBoostingTree(DataFrame trainDataArr, DataFrame trainLabelArr, NDArray D) {
        // 获得样本数目及特征数量
        int m = trainDataArr.nrow(), n = trainDataArr.ncol();
        // 单层树的字典，用于存放当前层提升树的参数
        // 也可以认为该字典代表了一层提升树
        SingleBoostTree sTree = new  SingleBoostTree();
        // 初始化分类误差率，分类误差率在算法8.1步骤（2）（b）有提到
        // 误差率最高也只能100%，因此初始化为1
        sTree.e = 1.0;

        double [] divs = {-0.5, 0.5, 1.5};
        String [] rules = {"LisOne", "HisOne"};

        // 对每一个特征进行遍历，寻找用于划分的最合适的特征
        for(int i : range(n)) {
            // 因为特征已经经过二值化，只能为0和1，因此分切分时分为-0.5， 0.5， 1.5三挡进行切割
            for (double div : divs) {
                // 在单个特征内对正反例进行划分时，有两种情况：
                // 可能是小于某值的为1，大于某值得为-1，也可能小于某值得是-1，反之为1
                // 因此在寻找最佳提升树的同时对于两种情况也需要遍历运行
                // LisOne：Low is one：小于某值得是1
                // HisOne：High is one：大于某值得是1
                for (String rule : rules) {
                    // 按照第i个特征，以值div进行切割，进行当前设置得到的预测和分类错误率
                    ArrayList<Integer> Gx = calc_e_Gx(trainDataArr, trainLabelArr, i, div, rule, D);
                    // 如果分类错误率e小于当前最小的e，那么将它作为最小的分类错误率保存

                    if( Tree_E < sTree.e ) {
                        sTree.e = Tree_E;
                        // 同时也需要存储最优划分点、划分规则、预测结果、特征索引
                        // 以便进行D更新和后续预测使用
                        sTree.div = div;
                        sTree.rule = rule;
                        sTree.feature = i;
                        sTree.Gx = Gx;
                    }
                }
            }
        }

        // 返回单层的提升树
        return sTree;
    }

    /**
     *     创建提升树
     *     创建算法依据“8.1.2 AdaBoost算法” 算法8.1
     *     @param trainDataList:训练数据集
     *     @param trainLabelList: 训练测试集
     *     @param treeNum: 树的层数
     *     :return: 提升树
     */
    public ArrayList<SingleBoostTree> createBosstingTree(DataFrame trainDataList,
                                                         DataFrame trainLabelList, int treeNum) {
        try (NDManager manager = NDManager.newBaseManager()) {
            // 将数据和标签转化为数组形式
            NDArray trainLabelArr = manager.create(trainLabelList.toMatrix().toArray());
            trainLabelArr = trainLabelArr.transpose();
            trainLabelArr.setRequiresGradient(false);

            // 没增加一层数后，当前最终预测结果列表
            double [] fdict = new double[trainLabelList.nrow()];
            Arrays.fill(fdict, 0);
            NDArray finalpredict = manager.create(fdict);
            finalpredict.setRequiresGradient(false);

            // 获得训练集数量以及特征个数
            int m = trainDataList.nrow(), n = trainDataList.ncol();

            // 依据算法8.1步骤（1）初始化D为1/N
            double [] Ddata = new double[m];
            Arrays.fill(Ddata, 1.0 / m);
            NDArray D = manager.create(Ddata);
            D.setRequiresGradient(false);

            // 初始化提升树列表，每个位置为一层
            ArrayList<SingleBoostTree> tree = new ArrayList<>();
            // 循环创建提升树
            for(int i : range(treeNum) ) {
                // 得到当前层的提升树
                SingleBoostTree curTree = createSingleBoostingTree(trainDataList, trainLabelList, D);
                // 根据式8.2计算当前层的alpha
                double alpha = 1.0 / 2 * Math.log((1 - curTree.e) / curTree.e);

                // 获得当前层的预测结果，用于下一步更新D
                ArrayList<Integer> Gd = curTree.Gx;
                NDArray Gx = manager.create(Gd.stream().mapToInt(k -> k).toArray());
                Gx = Gx.toType(DataType.FLOAT64, false);
                Gx.setRequiresGradient(false);

                // 依据式8.4更新D
                // 考虑到该式每次只更新D中的一个w，要循环进行更新知道所有w更新结束会很复杂（其实
                // 不是时间上的复杂，只是让人感觉每次单独更新一个很累），所以该式以向量相乘的形式，
                // 一个式子将所有w全部更新完。
                // 该式需要线性代数基础，如果不太熟练建议补充相关知识，当然了，单独更新w也一点问题没有
                // np.multiply(trainLabelArr, Gx)：exp中的y*Gm(x)，结果是一个行向量，内部为yi*Gm(xi)
                // np.exp(-1 * alpha * np.multiply(trainLabelArr, Gx))：上面求出来的行向量内部全体
                // 成员再乘以-αm，然后取对数，和书上式子一样，只不过书上式子内是一个数，这里是一个向量
                // D是一个行向量，取代了式中的wmi，然后D求和为Zm
                // 书中的式子最后得出来一个数w，所有数w组合形成新的D
                // 这里是直接得到一个向量，向量内元素是所有的w
                // 本质上结果是相同的

                //D = np.multiply(D, np.exp(-1 * alpha * np.multiply(trainLabelArr, Gx))) / sum(D)
                //System.out.println("===========================");
                //printVectorElements(trainLabelArr.toDoubleArray());
                //printVectorElements(Gx.toDoubleArray());
                NDArray T = trainLabelArr.mul(Gx);
                T = T.mul(-1 * alpha);

                double Dsum = D.sum().getDouble();
                D = D.mul(T.exp());
                D = D.div(Dsum);

                // 在当前层参数中增加alpha参数，预测的时候需要用到
                curTree.alpha = alpha;
                // 将当前层添加到提升树索引中。
                tree.add(curTree);

                // -----以下代码用来辅助，可以去掉---------------
                // 根据8 .6 式将结果加上当前层乘以α，得到目前的最终输出预测
                //finalpredict += alpha * Gx
                finalpredict = finalpredict.add( Gx.mul(alpha) );
                // 计算当前最终预测输出与实际标签之间的误差
                double error = 0.0;
                //sum([1 for i in range(len(trainDataList)) if np.sign(finalpredict[i]) != trainLabelArr[i]])
                for( int k : range(trainDataList.nrow()) ) {
                    if( sign(finalpredict.getDouble(k)) != trainLabelList.getInt(k, 0))
                        error += 1.0;
                }
                //System.out.println("error: "  + error);
                // 计算当前最终误差率
                double finalError = (1 * error / trainDataList.nrow());

                // 打印一些信息
                System.out.printf("Iter:%d:%d, sigle error:%.4f, final error:%.4f\n", i, treeNum, curTree.e, finalError);

                // 如果误差为0，提前退出即可，因为没有必要再计算算了
                if( finalError != 0 )
                    continue;
                else
                    return tree;
            }
            // 返回整个提升树
            return tree;
        }
    }

    /**
     *    输出单独层预测结果
     *     @param x: 预测样本
     *     @param div: 划分点
     *     @param rule: 划分规则
     *     @param feature: 进行操作的特征
     *     :return:
     */
    public int predict(DataFrame x, double div, String rule, int feature) {
        // 依据划分规则定义小于及大于划分点的标签
        int L = 0, H = 0;
        if( rule.equals("LisOne" ) ) {
            L = 1;
            H = -1;
        } else {
            L = -1;
            H = 1;
        }

        // 判断预测结果
        if( x.getInt(0, feature) < div )
            return L;
        else
            return H;
    }

    /**
     *     测试
     *     @param testDataList:测试数据集
     *     @param testLabelList: 测试标签集
     *     @param tree: 提升树
     *     @return: 准确率
     */
    public double model_test(DataFrame testDataList, DataFrame testLabelList, ArrayList<SingleBoostTree> tree) {
        // 错误率计数值
        int errorCnt = 0;
        // 遍历每一个测试样本
        for(int i : range(testDataList.nrow()) ) {
            // 预测结果值，初始为0
            double result = 0;
            // 依据算法8 .1 式8 .6
            // 预测式子是一个求和式，对于每一层的结果都要进行一次累加
            // 遍历每层的树
            for( SingleBoostTree curTree : tree ) {
                // 获取该层参数
                double div = curTree.div;
                String rule = curTree.rule;
                int feature = curTree.feature;
                double alpha = curTree.alpha;

                // 将当前层结果加入预测中
                result += alpha * predict(testDataList.of(i), div, rule, feature);
            }
            // 预测结果取sign值，如果大于0 sign为1，反之为0
            if( sign(result) != testLabelList.getInt(i, 0) )
                errorCnt += 1;
        }

        // 返回准确率
        return (1.0 - errorCnt*1.0 / testDataList.nrow())*100;
    }
    public static void main(String [] args) {
        Long start = System.currentTimeMillis();
        AdaBoost AB = new AdaBoost();

        String fnm = "./data/Mnist/mnist_train.csv";
        ArrayList<DataFrame> train_data = new ArrayList<>();
        AB.loadData(fnm, train_data);

        // 创建提升树
        System.out.println("start init train");
        ArrayList<SingleBoostTree> tree = AB.createBosstingTree(train_data.get(0).slice(0, 1000),
                train_data.get(1).slice(0, 1000), 50);
    /*
        for(SingleBoostTree t : tree) {
            System.out.printf("alpha: %.4f e: %.4f div: %.4f rule: %s feature: %d\n",
                    t.alpha, t.e, t.div, t.rule, t.feature);
        }

     */

        // 测试
        fnm = "./data/Mnist/mnist_test.csv";
        ArrayList<DataFrame> test_data = new ArrayList<>();
        AB.loadData(fnm, test_data);

        System.out.println("start to test");
        double accuracy = AB.model_test(test_data.get(0).slice(0, 100), test_data.get(1).slice(0, 100), tree);
        System.out.printf("The accuracy is: %.1f%s\n", accuracy, "%");

        System.out.printf("Time span: %ds\n", (System.currentTimeMillis() - start)/1000);
        System.exit(0);
    }
}
