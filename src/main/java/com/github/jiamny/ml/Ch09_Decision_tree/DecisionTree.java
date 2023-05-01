package com.github.jiamny.ml.Ch09_Decision_tree;

import com.github.jiamny.ml.utils.DataFrameHelper;
import com.github.jiamny.ml.utils.StatisticHelper;
import org.apache.commons.csv.CSVFormat;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import smile.data.DataFrame;
import smile.io.Read;
import smile.math.matrix.Matrix;

import java.util.*;

import static com.github.jiamny.ml.utils.DataFrameHelper.range;
import static com.github.jiamny.ml.utils.StatisticHelper.printVectorElements;

public class DecisionTree {

     /**
     *  建立字典，用于不同类别的标签技术
     *  * param labelArr: 标签集
     *  * return: 字典
     */
    public HashMap<Integer, Integer> labelMap(DataFrame labelArr) {
        // 建立字典，用于不同类别的标签技术
        HashMap<Integer, Integer> classDict = new HashMap<>();

        // 遍历所有标签
        for( int i = 0; i < labelArr.nrow(); i++ ) {
            // 当第一次遇到A标签时，字典内还没有A标签，这时候直接幅值加1是错误的，
            // 所以需要判断字典中是否有该键，没有则创建，有就直接自增
            if( classDict.isEmpty() ) {
                classDict.put((Integer)labelArr.get(i, 0), Integer.valueOf(1));
            } else {
                Integer key = (Integer)labelArr.get(i, 0);
                if( classDict.containsKey(key))   {
                    // 若在字典中存在该标签，则直接加1
                    int num = classDict.get((Integer)labelArr.get(i, 0)).intValue() + 1;
                    classDict.replace(key, Integer.valueOf(num));
                } else{
                    // 若无该标签，设初值为1，表示出现了1次了
                    classDict.put(key, Integer.valueOf(1));
                }
            }
        }
        return classDict;
    }

    /**
     * 找到当前标签集中占数目最大的标签
     * @param labelArr: 标签集
     * @return: 最大的标签
     */
    public int majorClass(DataFrame labelArr) {

        // 建立字典，用于不同类别的标签技术
        HashMap<Integer, Integer> classDict = labelMap(labelArr);

        // 占数目最多的标签
        Map.Entry<Integer, Integer> mxval = StatisticHelper.maxOrmin(classDict, Integer::compare, true);

        // 返回最大一项的标签，即占数目最多的标签
        //System.out.println(mxval.getValue().intValue());
        return mxval.getKey().intValue();
    }

    /**
     * 计算数据集D的经验熵，参考公式5.7 经验熵的计算
     * @param trainLabelArr 当前数据集的标签集
     * @return: 经验熵
     */
    public double calc_H_D(DataFrame trainLabelArr) {
        // 初始化为0
        double H_D = 0.0;
        /*
        将当前所有标签放入集合中，这样只要有的标签都会在集合中出现，且出现一次。遍历该集合就可以遍历所有出现过的标记
        并计算其Ck这么做有一个很重要的原因：首先假设一个背景，当前标签集中有一些标记已经没有了，比如说标签集中
        没有0（这是很正常的，说明当前分支不存在这个标签）。 式5.7中有一项Ck，那按照式中的针对不同标签k
        计算Cl和D并求和时，由于没有0，那么C0=0，此时C0/D0=0,log2(C0/D0) = log2(0)，事实上0并不在log的
        定义区间内，出现了问题 所以使用集合的方式先知道当前标签中都出现了那些标签，
        随后对每个标签进行计算，如果没出现的标签那一项就不在经验熵中出现（未参与，对经验熵无影响），
        保证log的计算能一直有定义
         */
        HashMap<Integer, Integer> classDict = labelMap(trainLabelArr);

        Set<Integer> trainLabelSet = classDict.keySet();
        Iterator it = trainLabelSet.iterator();
        // 遍历每一个出现过的标签
        while( it.hasNext() ) {
            Integer key = (Integer)it.next();
            // 计算|Ck|/|D|
            //trainLabelArr == i：当前标签集中为该标签的的位置
            //例如a = [1, 0, 0, 1], c = (a == 1): c == [True, false, false, True]
            //trainLabelArr[trainLabelArr == i]：获得为指定标签的样本
            //trainLabelArr[trainLabelArr == i].size：获得为指定标签的样本的大小，即标签为i的样本数量，就是|Ck|
            //trainLabelArr.size：整个标签集的数量（也就是样本集的数量），即|D|
            double p = classDict.get(key)*1.0 / trainLabelArr.nrow();
            //对经验熵的每一项累加求和
            H_D += -1 * p * Math.log(p);
        }
        //返回经验熵
        return H_D;
    }

    /**
     * 计算经验条件熵
     * @param trainDataArr_DevFeature 切割后只有feature那列数据的数组
     * @param trainLabelArr  标签集数组
     * @return: 经验条件熵
     */
    public double calcH_D_A(DataFrame trainDataArr_DevFeature, DataFrame trainLabelArr) {
        // 初始为0
        double H_D_A = 0;
        // 在featue那列放入集合中，是为了根据集合中的数目知道该feature目前可取值数目是多少
        HashMap<Integer, Integer> trainDict = labelMap(trainDataArr_DevFeature);
        Set<Integer> trainDataSet = trainDict.keySet();
        Iterator it = trainDataSet.iterator();

        //
        // 对于每一个特征取值遍历计算条件经验熵的每一项
        ArrayList<Integer> idx = new ArrayList<>();
        while(it.hasNext()) {
            // 计算H(D|A)
            // trainDataArr_DevFeature[trainDataArr_DevFeature == i].size / trainDataArr_DevFeature.size:|Di| / |D|
            // calc_H_D(trainLabelArr[trainDataArr_DevFeature == i]):H(Di)
            Integer key = (Integer) it.next();
            idx.clear();
            for( int i = 0; i < trainDataArr_DevFeature.nrow(); i++ ) {
                if( trainDataArr_DevFeature.getInt(i, 0) == key.intValue() )
                    idx.add(Integer.valueOf(i));
            }
            int [] sidx = idx.stream().mapToInt(i -> i).toArray();
            //printVectorElements(sidx);
            DataFrame sLabels = trainLabelArr.of(sidx);
            double d = trainDict.get(key).intValue()*1.0 / trainDataArr_DevFeature.nrow();
            H_D_A += d * calc_H_D(sLabels);
        }
        // 返回得出的条件经验熵
        return H_D_A;
    }

    /**
     *  计算信息增益最大的特征
     *  @param trainData: 当前数据集
     *  @param trainLabels: 当前标签集
     *  @return: 信息增益最大的特征及最大信息增益值
     */
    public ArrayList<String> calcBestFeature(DataFrame trainData, DataFrame trainLabels) {
        // 获取当前特征数目，也就是数据集的横轴大小
        int featureNum = trainData.ncol();

        // 初始化最大信息增益
        double maxG_D_A = -1;
        // 初始化最大信息增益的特征
        int maxFeature = -1;

        // “5.2.2 信息增益”中“算法5.1（信息增益的算法）”第一步：
        // 1.计算数据集D的经验熵H(D)
        double H_D = calc_H_D(trainLabels);

        // 对每一个特征进行遍历计算
        for( int fidx = 0; fidx < featureNum; fidx++ ) {
            // 2.计算条件经验熵H(D|A)
            // 由于条件经验熵的计算过程中只涉及到标签以及当前特征，为了提高运算速度（全部样本
            // 做成的矩阵运算速度太慢，需要剔除不需要的部分），将数据集矩阵进行切割
            // 数据集在初始时刻是一个Arr = 60000*784的矩阵，针对当前要计算的feature，在训练集中切割下
            // Arr[:, feature]这么一条来，因为后续计算中数据集中只用到这个（没明白的跟着算一遍例5.2）
            // trainData[:, feature]:在数据集中切割下这么一条
            // trainDataArr[:, feature].flat：将这么一条转换成竖着的列表
            // np.array(trainDataArr[:, feature].flat)：再转换成一条竖着的矩阵，大小为60000*1（只是初始是
            // 这么大，运行过程中是依据当前数据集大小动态变的）
            DataFrame trainDataArr_DevFeature = trainData.select(fidx);

            // 3.计算信息增益G(D|A)    G(D|A) = H(D) - H(D | A)
            double G_D_A = H_D - calcH_D_A(trainDataArr_DevFeature, trainLabels);

            // 不断更新最大的信息增益以及对应的feature
            if( G_D_A > maxG_D_A ) {
                maxG_D_A = G_D_A;
                maxFeature = fidx;
            }
        }
        ArrayList<String> rlt = new ArrayList<>();
        rlt.add(String.valueOf(maxFeature));
        rlt.add(String.valueOf(maxG_D_A));
        return rlt;
    }

    /**
     * 更新数据集和标签集
     * @param trainDataArr:要更新的数据集
     * @param trainLabelArr: 要更新的标签集
     * @param A: 要去除的特征索引
     * @param a: 当data[A]== a时，说明该行样本时要保留的
     * @return: 新的数据集和标签集
     */
    public ArrayList<DataFrame> getSubDataArr(DataFrame trainDataArr, DataFrame trainLabelArr, int A, Integer a) {
        // 返回的数据集
        DataFrame retDataArr = null;
        // 返回的标签集
        DataFrame retLabelArr = null;

        ArrayList<DataFrame> rlt = new ArrayList<>();

        ArrayList<Integer> cidx = new ArrayList<>();
        ArrayList<Integer> ridx = new ArrayList<>();
        // 对当前数据的每一个样本进行遍历
        for( int i = 0; i < trainDataArr.nrow(); i++ ) {
            // 如果当前样本的特征为指定特征值a
            if( trainDataArr.getInt(i, A) == a.intValue() ) {
                ridx.add(i);
            }
        }
        for( int c = 0; c < A; c++ )
            cidx.add(c);
        for( int c = A + 1; c < trainDataArr.ncol(); c++ )
            cidx.add(c);

        int [] rsidx = ridx.stream().mapToInt(i -> i).toArray();
        int [] csidx = cidx.stream().mapToInt(i -> i).toArray();

        // 那么将该样本的第A个特征切割掉，放入返回的数据集中
        retDataArr = trainDataArr.of(rsidx);
        rlt.add( retDataArr.select(csidx) );

        // 将该样本的标签放入返回标签集中
        rlt.add( trainLabelArr.of(rsidx) );

        // 返回新的数据集和标签集
        return rlt;
    }

    /**
     * 递归创建决策树
     * @param dataSet:(trainDataList， trainLabelList) <<-- 元祖形式
     * @return:新的子节点或该叶子节点的值
     */
    public DTNode  createTree( ArrayList<DataFrame> dataSet) {

        // 设置Epsilon，“5.3.1 ID3算法”第4步提到需要将信息增益与阈值Epsilon比较，若小于则
        // 直接处理后返回T
        // 该值的大小在设置上并未考虑太多，观察到信息增益前期在运行中为0.3左右，所以设置了0.1
        double Epsilon = 0.1;
        // 从参数中获取trainDataList和trainLabelList
        // 之所以使用元祖作为参数，是由于后续递归调用时直数据集需要对某个特征进行切割，在函数递归
        // 调用上直接将切割函数的返回值放入递归调用中，而函数的返回值形式是元祖的，等看到这个函数
        // 的底部就会明白了，这样子的用处就是写程序的时候简洁一点，方便一点
        DataFrame trainDataList = dataSet.get(0);
        DataFrame  trainLabelList = dataSet.get(1);
        // 打印信息：开始一个子节点创建，打印当前特征向量数目及当前剩余样本数目
        System.out.println("start a node " + trainDataList.ncol() + " " + trainLabelList.nrow());

        // 将标签放入一个字典中，当前样本有多少类，在字典中就会有多少项
        // 也相当于去重，多次出现的标签就留一次。举个例子，假如处理结束后字典的长度为1，那说明所有的样本
        // 都是同一个标签，那就可以直接返回该标签了，不需要再生成子节点了。
        //classDict = {i for i in trainLabelList}
        HashMap<Integer, Integer> classDict =  labelMap(trainLabelList);

        // 如果D中所有实例属于同一类Ck，则置T为单节点数，并将Ck作为该节点的类，返回T
        // 即若所有样本的标签一致，也就不需要再分化，返回标记作为该节点的值，返回后这就是一个叶子节点
        if( classDict.size() == 1 ) {
            // 因为所有样本都是一致的，在标签集中随便拿一个标签返回都行，这里用的第0个（因为你并不知道
            // 当前标签集的长度是多少，但运行中所有标签只要有长度都会有第0位。
            DTNode new_node = new DTNode();
            new_node.leaf = true;
            new_node.label = trainLabelList.getInt(0, 0);
            return new_node;
        }

        // 如果A为空集，则置T为单节点数，并将D中实例数最大的类Ck作为该节点的类，返回T
        // 即如果已经没有特征可以用来再分化了，就返回占大多数的类别
        if( trainDataList.nrow() == 0 ) {
            // 返回当前标签集中占数目最大的标签
            DTNode new_node = new DTNode();
            new_node.leaf = true;
            new_node.label = majorClass(trainLabelList);
            return new_node;
        }

        // 否则，按式5.10计算A中个特征值的信息增益，选择信息增益最大的特征Ag
        ArrayList<String> rlt = calcBestFeature(trainDataList, trainLabelList);
        int Ag = Integer.parseInt(rlt.get(0));
        double EpsilonGet = Double.parseDouble(rlt.get(1));

        // 如果Ag的信息增益比小于阈值Epsilon，则置T为单节点树，并将D中实例数最大的类Ck
        // 作为该节点的类，返回T
        if( EpsilonGet < Epsilon ) {
            //return majorClass(trainLabelList);
            DTNode new_node = new DTNode();
            new_node.leaf = true;
            new_node.label = majorClass(trainLabelList);
            return new_node;
        }

        // 否则，对Ag的每一可能值ai，依Ag=ai将D分割为若干非空子集Di，将Di中实例数最大的
        // 类作为标记，构建子节点，由节点及其子节点构成树T，返回T
        DTNode rootNode = new DTNode();
        rootNode.leaf = false;
        //rootNode.label = Ag;
        rootNode.attribute = Ag;
        rootNode.threshold = EpsilonGet;
        // 特征值为0时，进入0分支
        // getSubDataArr(trainDataList, trainLabelList, Ag, 0)：在当前数据集中切割当前feature，返回新的数据集和标签集
        rootNode.left = createTree(getSubDataArr(trainDataList, trainLabelList, Ag, 0));
        rootNode.right = createTree(getSubDataArr(trainDataList, trainLabelList, Ag, 1));
        return rootNode;
    }

    //输出决策树
    public void printDTree(DTNode node) {
        if(node.left==null && node.right == null){
            System.out.println("-- " + node.label + " " + node.attribute);
            return;
        } else {
            if( node.left != null ) {
                System.out.println("== " + node.label + " " + node.attribute);
                printDTree(node.left);
            }

            if( node.right != null ) {
                System.out.println("++ " + node.label + " " + node.attribute);
                printDTree(node.right);
            }
        }
    }

    /**
     * 预测标签
     *  @param testDataList:样本
     *  @param tree: 决策树
     *  @return: 预测结果
     */
    public int predict(DataFrame testDataList, DTNode tree) {

        // 死循环，直到找到一个有效地分类
        while(true) {
            // 因为有时候当前字典只有一个节点
            // 例如{73: {0: {74:6}}}看起来节点很多，但是对于字典的最顶层来说，只有73一个key，其余都是value
            // 若还是采用for来读取的话不太合适，所以使用下行这种方式读取key和value

            if( tree.left == null && tree.right == null ) {
                return tree.label;
            } else {
                int feature = tree.attribute;
                int dataVal  = testDataList.getInt(0, feature);

                // 获取目前所在节点的feature值，需要在样本中删除该feature
                // 因为在创建树的过程中，feature的索引值永远是对于当时剩余的feature来设置的
                // 所以需要不断地删除已经用掉的特征，保证索引相对位置的一致性
                testDataList = testDataList.drop(feature);

                // 将tree更新为其子节点
                if( dataVal == 0 ) {
                    tree = tree.left;
                } else {
                    tree = tree.right;
                }

                if( tree.leaf )
                    return tree.label;
            }
        }
    }

    /**
     * 测试准确率
     * @param testDataList:待测试数据集
     * @param testLabelList: 待测试标签集
     * @param tree: 训练集生成的树
     * @return: 准确率
     */
    public double model_test(DataFrame testDataList, DataFrame testLabelList, DTNode tree) {
        // 错误次数计数
        int errorCnt = 0;
        int n_sample = testDataList.nrow();
        // 遍历测试集中每一个测试样本
        for( int i = 0; i < n_sample; i++ ) {
            // 判断预测与标签中结果是否一致
            if( testLabelList.getInt(i, 0) != predict( testDataList.of(i), tree) )
                errorCnt += 1;
        }

        // 返回准确率
        return 1.0 - (errorCnt*1.0 / n_sample);
    }

    public static void main(String [] args) {
        DecisionTree DT = new DecisionTree();
        String fnm = "./data/Mnist/mnist_train.csv";
        ArrayList<DataFrame> train_data = new ArrayList<>();

        DataFrameHelper.loadMnistData(fnm, train_data);
        DTNode tree = DT.createTree( train_data );
        //DT.printDTree(tree);

        fnm = "./data/Mnist/mnist_test.csv";
        ArrayList<DataFrame> test_data = new ArrayList<>();
        DataFrameHelper.loadMnistData(fnm, test_data);

        System.out.println("The accur is: " + DT.model_test( test_data.get(0), test_data.get(1), tree));
        System.exit(0);
    }
}
