package com.github.jiamny.ml.Ch13_ExpectationMaximum;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import smile.stat.distribution.*;

import java.util.*;

import static com.github.jiamny.ml.utils.StatisticHelper.ShuffleArray;

public class ExpectationMaximization {
    /**
     *     初始化数据集
     *     这里通过服从高斯分布的随机函数来伪造数据集
     *     @param mu0: 高斯0的均值
     *     @param sigma0: 高斯0的方差
     *     @param mu1: 高斯1的均值
     *     @param sigma1: 高斯1的方差
     *     @param alpha0: 高斯0的系数
     *     @param alpha1: 高斯1的系数
     *     @return: 混合了两个高斯分布的数据
     */
    public double [] loadData(double mu0, double sigma0, double mu1, double sigma1, double alpha0, double alpha1) {
        // 定义数据集长度为1000
        int length = 1000;

        // 初始化第一个高斯分布，生成数据，数据长度为length * alpha系数，以此来
        // 满足alpha的作用
        GaussianDistribution gg = new GaussianDistribution(mu0, sigma0);
        double [] d1 = gg.rand((int)( length*alpha0));
        //data0 = np.random.normal(mu0, sigma0, int(length * alpha0))
        // 第二个高斯分布的数据
        gg = new GaussianDistribution(mu1, sigma1);
        double [] d2 = gg.rand((int)( length*alpha1));
        //data1 = np.random.normal(mu1, sigma1, int(length * alpha1))

        // 初始化总数据集
        //两个高斯分布的数据混合后会放在该数据集中返回
        //dataSet = []
        double [] dataSet;

        // 将第一个数据集的内容添加进去
        //dataSet.extend(data0)
        dataSet = Arrays.copyOf(d1, d1.length + d2.length);
        // 添加第二个数据集的数据
        //dataSet.extend(data1)
        System.arraycopy(d2, 0, dataSet, d1.length, d2.length);
        // 对总的数据集进行打乱（其实不打乱也没事，只不过打乱一下直观上让人感觉已经混合了
        //  读者可以将下面这句话屏蔽以后看看效果是否有差别）
        // random.shuffle(dataSet);
        dataSet = ShuffleArray(dataSet);

        // 返回伪造好的数据集
        return dataSet;
    }

    /**
     *     根据高斯密度函数计算值
     *     依据：“9.3.1 高斯混合模型” 式9.25
     *     注：在公式中y是一个实数，但是在EM算法中(见算法9.2的E步)，需要对每个j
     *     都求一次yjk，在本实例中有1000个可观测数据，因此需要计算1000次。考虑到
     *     在E步时进行1000次高斯计算，程序上比较不简洁，因此这里的y是向量，在numpy
     *     的exp中如果exp内部值为向量，则对向量中每个值进行exp，输出仍是向量的形式。
     *     所以使用向量的形式1次计算即可将所有计算结果得出，程序上较为简洁
     *     @param dataSetArr: 可观测数据集
     *     @param mu: 均值
     *     @param sigmod: 方差
     *     @return: 整个可观测数据集的高斯分布密度（向量形式）
     */
    public NDArray calcGauss(NDArray dataSetArr, double mu, double sigmod) {
        // 计算过程就是依据式9.25写的，没有别的花样
        // (1 / (math.sqrt(2 * math.pi) * sigmod)) * np.exp(-1 * (dataSetArr - mu) * (dataSetArr - mu) / (2 * sigmod**2));
        double sqtResult = (1 / (Math.sqrt(2 * Math.PI) * sigmod));

        NDArray dif = dataSetArr.sub(mu);
        NDArray difp = ((dif.mul(dif)).mul(-1).div(Math.pow(sigmod, 2)*2)).exp();
        NDArray expResult = difp.mul(sqtResult);
        // 返回结果
        return expResult;
    }

    /**
     *    EM算法中的E步
     *    依据当前模型参数，计算分模型k对观数据y的响应度
     *    @param dataSetArr: 可观测数据y
     *    @param alpha0: 高斯模型0的系数
     *    @param mu0: 高斯模型0的均值
     *    @param sigmod0: 高斯模型0的方差
     *    @param alpha1: 高斯模型1的系数
     *    @param mu1: 高斯模型1的均值
     *    @param sigmod1: 高斯模型1的方差
     *    @return: 两个模型各自的响应度
     */
    public ArrayList<NDArray> E_step(NDArray dataSetArr, double alpha0, double mu0, double sigmod0,
                  double alpha1, double mu1, double sigmod1) {
        ArrayList<NDArray> rlt = new ArrayList<>();

        // 计算y0的响应度
        // 先计算模型0的响应度的分子
        NDArray gamma0 = calcGauss(dataSetArr, mu0, sigmod0).mul(alpha0);
        // 模型1响应度的分子
        NDArray gamma1 = calcGauss(dataSetArr, mu1, sigmod1).mul(alpha1);

        // 两者相加为E步中的分布
        NDArray sum = gamma0.add(gamma1);
        // 各自相除，得到两个模型的响应度
        gamma0 = gamma0.div( sum );
        gamma1 = gamma1.div( sum );

        //返回两个模型响应度
        rlt.add(gamma0);
        rlt.add(gamma1);
        return rlt;
    }

    /**
     *
     */
    public ArrayList<Double> M_step(double muo, double mu1,
            NDArray gamma0, NDArray gamma1, NDArray dataSetArr) {
        ArrayList<Double> rlt = new ArrayList<>();

        // 依据算法9.2计算各个值
        // 这里没什么花样，对照书本公式看看这里就好了
        double mu0_new = ((gamma0.dot(dataSetArr)).div(gamma0.sum())).toDoubleArray()[0];
        double mu1_new = ((gamma1.dot(dataSetArr)).div(gamma1.sum())).toDoubleArray()[0];

        double sigmod0_new = Math.sqrt((gamma0.dot(dataSetArr.sub(muo).pow(2))).toDoubleArray()[0]/gamma0.sum().toDoubleArray()[0]);
        double sigmod1_new = Math.sqrt((gamma1.dot((dataSetArr.sub(mu1)).pow(2))).toDoubleArray()[0] / gamma1.sum().toDoubleArray()[0]);

        double alpha0_new = gamma0.sum().toDoubleArray()[0] / gamma0.size();
        double alpha1_new = gamma1.sum().toDoubleArray()[0] / gamma1.size();

        // 将更新的值返回
        rlt.add(mu0_new);
        rlt.add(mu1_new);
        rlt.add(sigmod0_new);
        rlt.add(sigmod1_new);
        rlt.add(alpha0_new);
        rlt.add(alpha1_new);
        return rlt;
    }

    /**
     *    根据EM算法进行参数估计
     *    算法依据“9.3.2 高斯混合模型参数估计的EM算法” 算法9.2
     *    @param dataSetList:数据集（可观测数据）
     *    @param iter: 迭代次数
     *    @return: 估计的参数
     */
    public ArrayList<Double> EM_Train(double [] dataSetList, int iter ) {

        try (NDManager manager = NDManager.newBaseManager()) {
            ArrayList<Double> rlt = new ArrayList<>();

            // 将可观测数据y转换为数组形式，主要是为了方便后续运算
            NDArray dataSetArr = manager.create(dataSetList);

            // 步骤1：对参数取初值，开始迭代
            double alpha0 = 0.5, mu0 = 0, sigmod0 = 1;
            double alpha1 = 0.5, mu1 = 1, sigmod1 = 1;

            // 开始迭代
            int step = 0;
            while (step < iter) {
                // 每次进入一次迭代后迭代次数加1
                step += 1;
                // 步骤2：E步：依据当前模型参数，计算分模型k对观测数据y的响应度
                ArrayList<NDArray> gammas = E_step(dataSetArr, alpha0, mu0, sigmod0, alpha1, mu1, sigmod1);
                NDArray gamma0 = gammas.get(0), gamma1 = gammas.get(1);

                // 步骤3：M步
                ArrayList<Double> nums = M_step(mu0, mu1, gamma0, gamma1, dataSetArr);
                mu0 = nums.get(0);
                mu1 = nums.get(1);
                sigmod0 = nums.get(2);
                sigmod1 = nums.get(3);
                alpha0 = nums.get(4);
                alpha1 = nums.get(5);
                System.out.printf("M step iter:%4d, alpha0: %.1f, mu0: %.1f, sigmod0: %.1f, alpha1: %.1f, mu1: %.1f, sigmod1: %.1f\n",
                        step, alpha0, mu0, sigmod0, alpha1, mu1, sigmod1);
            }
            // 迭代结束后将更新后的各参数返回
            rlt.add(alpha0);
            rlt.add(mu0);
            rlt.add(sigmod0);
            rlt.add(alpha1);
            rlt.add(mu1);
            rlt.add(sigmod1);
            return rlt;
        }
    }

    public static void main(String [] args) {

        Long start = System.currentTimeMillis();
        ExpectationMaximization EM = new ExpectationMaximization();

        // 设置两个高斯模型进行混合，这里是初始化两个模型各自的参数
        // 见“9.3 EM算法在高斯混合模型学习中的应用”
        // alpha是“9.3.1 高斯混合模型” 定义9.2中的系数α
        // mu0是均值μ
        // sigmod是方差σ
        // 在设置上两个alpha的和必须为1，其他没有什么具体要求，符合高斯定义就可以
        double alpha0 = 0.3, mu0 = -2, sigmod0 = 0.5;
        double alpha1 = 0.7, mu1 = 0.5, sigmod1 = 1;
        double[] data = EM.loadData(mu0, sigmod0, mu1, sigmod1, alpha0, alpha1);

        // 打印参数预测结果
        System.out.println("----------------------------");
        System.out.println("the Parameters set is::");
        System.out.printf("alpha0: %.1f, mu0: %.1f, sigmod0: %.1f, alpha1: %.1f, mu1: %.1f, sigmod1: %.1f\n",
                        alpha0, mu0, sigmod0, alpha1, mu1, sigmod1);

        // 开始EM算法，进行参数估计
        ArrayList<Double> rlt = EM.EM_Train(data, 500);
        alpha0  = rlt.get(0);
        mu0     = rlt.get(1);
        sigmod0 = rlt.get(2);
        alpha1  = rlt.get(3);
        mu1     = rlt.get(4);
        sigmod1 = rlt.get(5);

        // 打印参数预测结果
        System.out.println("----------------------------");
        System.out.println("the Parameters predict is:");
        System.out.printf("alpha0: %.1f, mu0: %.1f, sigmod0: %.1f, alpha1: %.1f, mu1: %.1f, sigmod1: %.1f\n",
                alpha0, mu0, sigmod0, alpha1, mu1, sigmod1);

        // 打印时间
        System.out.println("----------------------------");
        System.out.printf("Time span: %ds\n", (System.currentTimeMillis() - start)/1000);
        System.exit(0);
    }
}
