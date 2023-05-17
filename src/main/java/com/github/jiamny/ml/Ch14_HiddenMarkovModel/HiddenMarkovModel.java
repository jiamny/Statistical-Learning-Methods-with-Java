package com.github.jiamny.ml.Ch14_HiddenMarkovModel;

import com.github.jiamny.ml.utils.StatisticHelper;
import smile.data.DataFrame;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;

import static com.github.jiamny.ml.utils.DataFrameHelper.range;
import static com.github.jiamny.ml.utils.StatisticHelper.printVectorElements;

public class HiddenMarkovModel {

    /**
     *     依据训练文本统计PI、A、B
     *     @param fileName: 训练文本
     *     @return: 三个参数
     */
    public ArrayList<DataFrame> trainParameter(String fileName) {

        // 定义一个查询字典，用于映射四种标记在数组中对应的位置，方便查询
        // B：词语的开头
        // M：一个词语的中间词
        // E：一个词语的结果
        // S：非词语，单个词
        // {'B':0, 'M':1, 'E':2, 'S':3}
        HashMap<String, Integer> statuDict = new HashMap<>();
        statuDict.put("B", 0);
        statuDict.put("M", 1);
        statuDict.put("E", 2);
        statuDict.put("S", 3);

        // 每个字只有四种状态，所以下方的各类初始化中大小的参数均为4
        // 初始化PI的一维数组，因为对应四种状态，大小为4
        double [][] PIdata = new double[1][4];

        // 初始化状态转移矩阵A，涉及到四种状态各自到四种状态的转移，因为大小为4x4
        double [][] Adata = new double[4][4];
            for(int i : range(4))
                Arrays.fill(Adata[i], 0);

        // 初始化观测概率矩阵，分别为四种状态到每个字的发射概率
        // 因为是中文分词，使用ord(汉字)即可找到其对应编码，这里用一个65536的空间来保证对于所有的汉字都能
        // 找到对应的位置来存储

        double [][] Bdata  = new double[4][65536];
        for(int i : range(4))
            Arrays.fill(Bdata[i], 0);

        // 去读训练文本
        // open(fileName, encoding='utf-8')
        File fr = new File(fileName);
        BufferedReader in = null;

        // 文本中的每一行认为是一个训练样本
        // 在统计上，三个参数依据“10.3.2” Baum-Welch算法内描述的统计
        // PI依据式10.35
        // A依据10.37
        // B依据10.38
        // 注：并没有使用Baum-Welch算法，只是借助了其内部的三个参数生成公式，其实
        // 公式并不是Baum-Welch特有的，只是在那一节正好有描述
        try {
            in = new BufferedReader( new InputStreamReader(new FileInputStream(fr), "UTF-8"));
            String line = "";
            while( (line = in.readLine()) != null) {
                // ---------------------训练集单行样例--------------------
                // 深圳  有  个  打工者  阅览室
                // ------------------------------------------------------
                // 可以看到训练样本已经分词完毕，词语之间空格隔开，因此我们在生成统计时主要借助以下思路：
                // 1.先将句子按照空格隔开，例如例句中5个词语，隔开后变成一个长度为5的列表，每个元素为一个词语
                // 2.对每个词语长度进行判断：
                //       如果为1认为该词语是S，即单个字
                //       如果为2则第一个是B，表开头，第二个为E，表结束
                //       如果大于2，则第一个为B，最后一个为E，中间全部标为M，表中间词
                // 3.统计PI：该句第一个字的词性对应的PI中位置加1
                //           例如：PI = [0， 0， 0， 0]，当本行第一个字是B，即表示开头时，PI中B对应位置为0，
                //           则PI = [1， 0， 0， 0]，全部统计结束后，按照计数值再除以总数得到概率
                //   统计A：对状态链中位置t和t-1的状态进行统计，在矩阵中相应位置加1，全部结束后生成概率
                //   统计B：对于每个字的状态以及字内容，生成状态到字的发射计数，全部结束后生成概率
                //   注：可以看一下“10.1.1 隐马尔可夫模型的定义”一节中三个参数的定义，会有更清晰一点的认识
                // -------------------------------------------------------
                // 对单行句子按空格进行切割
                String [] curLine = line.strip().split(" ");

                // 对词性的标记放在该列表中
                ArrayList<String> wordLabel = new ArrayList<>();

                // 对每一个单词进行遍历
                for(int i : range(curLine.length)) {

                    if( curLine[i].length() < 1 )
                        continue;

                    String label = "";
                    // 如果长度为1，则直接将该字标记为S，即单个词
                    if (curLine[i].length() == 1)
                        label = "S";
                    else {
                        // 如果长度不为1，开头为B，最后为E，中间添加长度-2个M
                        // 如果长度刚好为2，长度-2=0也就不添加了，反之添加对应个数的M
                        label = "B";
                        for (int j = 0; j < (curLine[i].length() - 2); j++)
                            label += "M";
                        label += "E";
                    }

                    // 如果是单行开头第一个字，PI中对应位置加1,
                    if (i == 0) {
                        //long v = PI.getLong(statuDict.get(String.valueOf(label.charAt(0)))) + 1;
                        //PI.set(new NDIndex(statuDict.get(String.valueOf(label.charAt(0)))), v);
                        PIdata[0][statuDict.get(String.valueOf(label.charAt(0)))] += 1;
                    }

                    // 对于该单词中的每一个字，在生成的状态链中统计B
                    //printVectorElements(range(label.length()));
                    for (int j : range(label.length())) {
                        // 遍历状态链中每一个状态，并找到对应的中文汉字，在B中
                        // 对应位置加1
                        //B[statuDict[label[j]]][ord(curLine[i][j])] += 1;
                        int r = statuDict.get(String.valueOf(label.charAt(j)));
                        int c = (int) curLine[i].charAt(j);
                        Bdata[r][c] += 1;
                    }

                    // 在整行的状态链中添加该单词的状态链
                    // 注意：extend表直接在原先元素的后方添加，
                    // 可以百度一下extend和append的区别
                    //wordLabel.extend(label);
                    if( ! label.equalsIgnoreCase("") ) {
                        //System.out.println("label " + label);
                        wordLabel.add(String.valueOf(label.charAt(0)));
                    }
                }
                // 单行所有单词都结束后，统计A信息
                // 因为A涉及到前一个状态，因此需要等整条状态链都生成了才能开始统计
                for(int i = 1;  i < wordLabel.size(); i++) {
                    // 统计t时刻状态和t-1时刻状态的所有状态组合的出现次数
                    Adata[statuDict.get(wordLabel.get(i - 1))][statuDict.get(wordLabel.get(i))] += 1;
                }
            }
            in.close();
        } catch(Exception e) {
            e.printStackTrace();
        }

        // 上面代码在统计上全部是统计的次数，实际运算需要使用概率，
        // 下方代码是将三个参数的次数转换为概率
        // ----------------------------------------
        // 对PI求和，概率生成中的分母
        //sum = np.sum(PI)
        double sum = StatisticHelper.sum(PIdata[0]);
        //for(int i = 0; i < PIdata[0].length; i++)
        //    sum += PIdata[0][i];

        // 遍历PI中每一个元素，元素出现的次数/总次数即为概率
        for(int i : range(PIdata[0].length) ) {
            // 如果某元素没有出现过，该位置为0，在后续的计算中这是不被允许的
            // 比如说某个汉字在训练集中没有出现过，那在后续不同概率相乘中只要有
            // 一项为0，其他都是0了，此外整条链很长的情况下，太多0-1的概率相乘
            // 不管怎样最后的结果都会很小，很容易下溢出
            // 所以在概率上我们习惯将其转换为log对数形式，这在书上是没有讲的
            // x大的时候，log也大，x小的时候，log也相应小，我们最后比较的是不同
            // 概率的大小，所以使用log没有问题

            // 那么当单向概率为0的时候，log没有定义，因此需要单独判断
            // 如果该项为0，则手动赋予一个极小值
            if (PIdata[0][i] == 0)
                PIdata[0][i] = -3.14e+100;
            // 如果不为0，则计算概率，再对概率求log
            else
                PIdata[0][i] = Math.log(PIdata[0][i] / sum);
        }
        // 与上方PI思路一样，求得A的概率对数
        for( int i : range(Adata.length) ) {
            sum = StatisticHelper.sum(Adata[i]);
            for( int j : range(Adata[i].length) ) {
                if( Adata[i][j] == 0.0 )
                    Adata[i][j] = -3.14e+100;
                else
                    Adata[i][j] = Math.log(Adata[i][j] / sum);
            }
        }

        // 与上方PI思路一样，求得B的概率对数
        for(int i : range(Bdata.length) ) {
            sum = StatisticHelper.sum(Bdata[i]);
            for(int j : range(Bdata[i].length) ) {
                if( Bdata[i][j] == 0.0 )
                    Bdata[i][j] = -3.14e+100;
                else
                    Bdata[i][j] = Math.log(Bdata[i][j] / sum);
            }
        }

        // 返回统计得到的三个参数
        ArrayList<DataFrame> rlt = new ArrayList<>();
        rlt.add(DataFrame.of(PIdata));
        rlt.add(DataFrame.of(Adata));
        rlt.add(DataFrame.of(Bdata));
        return rlt;
    }

    /**
     *    加载文章
     *    @param fileName:文件路径
     *    @return: 文章内容
     */
    public ArrayList<String> loadArticle(String fileName) {
        File fr = new File(fileName);
        BufferedReader in = null;
        ArrayList<String> articles = new ArrayList<>();
        try {
            in = new BufferedReader(new InputStreamReader(new FileInputStream(fr), "UTF-8"));
            String line = "";
            // 按行读取文件
            while ((line = in.readLine()) != null) {
                line.replaceAll("\n|\r", "");
                // 将该行放入文章列表中
                articles.add(line);
            }
            in.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        // 将文章返回
        return articles;
    }

    /**
     *     分词
     *     算法依据“10.4.2 维特比算法”
     *     @param article:要分词的文章
     *     @param PI: 初始状态概率向量PI
     *     @param A: 状态转移矩阵
     *     @param B: 观测概率矩阵
     *     @return: 分词后的文章
     */
    public ArrayList<String> participle(ArrayList<String> article, double [][] PI, double [][] A, double [][] B) {

        // 初始化分词后的文章列表
        ArrayList<String> retArtical = new ArrayList<>();

        // 对文章按行读取
        for( String line : article ) {
            // 初始化δ，δ存放四种状态的概率值，因为状态链中每个状态都有
            // 四种概率值，因此长度时该行的长度
            //delta = [[0 for i in range(4)] for i in range(len(line))]
            double [][] delta = new double[line.length()][4];

            // 依据算法10.5 第一步：初始化
            for(int i : range(4) ) {
                // 初始化δ状态链中第一个状态的四种状态概率
                int c = (int) line.charAt(0);
                delta[0][i] = PI[0][i] + B[i][c]; //ord(line[0])];
            }

            // 初始化ψ，初始时为0
            //psi = [[0 for i in range(4)] for i in range(len(line))]
            int [][] psi = new int[line.length()][4];

            // 算法10.5中的第二步：递推
            // for循环的符号与书中公式一致，可以对比着看来理解
            // 依次处理整条链
            for( int t = 1; t < line.length(); t++ ) {
                //对于链中的米格状态，求四种状态概率
                for( int i : range(4) ) {
                    // 初始化一个临时列表，用于存放四种概率
                    // tmpDelta = [0] *4
                    double [] tmpDelta = new double[4];
                    Arrays.fill(tmpDelta, 0);

                    for(int j : range(4) ) {
                        // 计算第二步中的δ，该部分只计算max内部，不涉及后面的bi(o)
                        // 计算得到四个结果以后，再去求那个max即可
                        // 注：bi(Ot) 并不在max的式子中，是求出max以后再乘b的
                        // 此外读者可能注意到书中的乘法在这里变成了加法，这是由于原先是概率
                        // 直接相乘，但我们在求得概率时，同时取了log，取完log以后，概率的乘法
                        // 也就转换为加法了，同时也简化了运算
                        // 所以log优点还是很多的对不？
                        tmpDelta[j] = delta[t - 1][j] + A[j][i];
                    }

                    // 记录最大值对应的状态
                    int maxDeltaIndex = StatisticHelper.maxIndex(tmpDelta); //tmpDelta.index(maxDelta)

                    // 找到最大的那个δ * a，
                    double maxDelta = tmpDelta[maxDeltaIndex];

                    // 将找到的最大值乘以b放入，
                    // 注意：这里同样因为log变成了加法
                    int c = (int) line.charAt(t);
                    delta[t][i] = maxDelta + B[i][c];   // ord(line[t])]
                    // 在ψ中记录对应的最大状态索引
                    psi[t][i] = maxDeltaIndex;
                }
            }

            // 建立一个状态链列表，开始生成状态链
            ArrayList<Integer> sequence = new ArrayList<>();
            // 算法10.5 第三步：终止
            // 在上面for循环全部结束后，很明显就到了第三步了
            // 获取最后一个状态的最大状态概率对应的索引
            //i_opt = delta[len(line) - 1].index(max(delta[len(line) - 1]))
            int i_opt = StatisticHelper.maxIndex(delta[line.length() - 1]);

            // 在状态链中添加索引
            // 注：状态链应该是B、M、E、S，这里图方便用了0、1、2、3，其实一样的
            sequence.add(i_opt);

            // 算法10.5 第四步：最优路径回溯
            // 从后往前遍历整条链
            for(int t = line.length() - 1; t > 0; t-- ){
                // 不断地从当前时刻t的ψ列表中读取到t-1的最优状态
                i_opt = psi[t][i_opt];
                // 将状态放入列表中
                sequence.add(i_opt);
            }

            // 因为是从后往前将状态放入的列表，所以这里需要翻转一下，变成了从前往后
            //sequence.reverse();
            Collections.reverse(sequence);

            // 开始对该行分词
            String curLine = "";
            // 遍历该行每一个字
            for(int i : range(line.length())) {
                // 在列表中放入该字
                curLine += String.valueOf( line.charAt(i) );
                // 如果该字是3：S -> 单个词 或 2:E -> 结尾词 ，则在该字后面加上分隔符 |
                // 此外如果改行的最后一个字了，也就不需要加 |
                if((sequence.get(i) == 3 || sequence.get(i) ==2) && i != (line.length() - 1) )
                    curLine += "|";
            }
            // 在返回列表中添加分词后的该行
            retArtical.add(curLine);
        }

        // 返回分词后的文章
        return retArtical;
    }

    public static void main(String [] args) {
        Long start = System.currentTimeMillis();
        HiddenMarkovModel HMM = new HiddenMarkovModel();

        // 依据现有训练集统计PI、A、B
        ArrayList<DataFrame> PAB = HMM.trainParameter("data/HMMTrainSet.txt");
        double [][] PI = PAB.get(0).toMatrix().toArray();
        double [][] A = PAB.get(1).toMatrix().toArray();
        double [][] B = PAB.get(2).toMatrix().toArray();

        // 读取测试文章
        ArrayList<String> article = HMM.loadArticle("data/testArtical.txt");
        // 打印原文
        System.out.println("-------------------原文----------------------");
        for( String line : article )
            System.out.println(line);

        // 进行分词
        ArrayList<String> partiArtical = HMM.participle(article, PI, A, B);

        // 打印分词结果
        System.out.println("\n-------------------分词后----------------------");
        for( String line : partiArtical )
            System.out.println(line);

        // 打印时间
        System.out.println("----------------------------");
        System.out.printf("Time span: %ds\n", (System.currentTimeMillis() - start)/1000);
        System.exit(0);
    }
}
