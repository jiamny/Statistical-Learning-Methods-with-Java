package com.github.jiamny.ml.Ch22_MarkovChainMonteCarlo;

import smile.math.Random;
import smile.math.matrix.Matrix;
import smile.stat.distribution.MultivariateGaussianDistribution;

import java.util.ArrayList;
import java.util.Arrays;
import static com.github.jiamny.ml.utils.DataFrameHelper.range;
import static com.github.jiamny.ml.utils.StatisticHelper.printVectorElements;

public class GibbsSamplingMethod {

    //目标求均值函数
    public double f( double [] x) {
        return x[0] + x[1];
    }

    /*
    吉布斯抽样算法在二元正态分布中抽取样本

    与np.random.multivariate_normal方法类似

    :param mean: n元正态分布的均值
    :param cov: n元正态分布的协方差矩阵
    :param n_samples: 样本量
    :param m: 收敛步数
    :param random_state: 随机种子
    :return: 随机样本列表
     */
    public double gibbs_sampling_method(double [] mean , double [][] cov, int n_samples, int m,
                                        long random_state, ArrayList<double []> samples) {
        Random rd = new Random(random_state);

        // 选取初始样本
        double [] x0 = Arrays.copyOf(mean, mean.length);

        double sum_ = 0;    // 目标求均值函数的和

        // 循环执行n次迭代
        for( int k : range(m + n_samples) ) {
            // 根据满条件分布逐个抽取样本
            //x0[0] = np.random.multivariate_normal([x0[1] * cov[0][1]], np.diag([1 - pow(cov[0][1], 2)]),1)[0][0]
            //x0[1] = np.random.multivariate_normal([x0[0] * cov[0][1]], np.diag([1 - pow(cov[0][1], 2)]),1)[0][0]

            x0[0] = (new MultivariateGaussianDistribution(
                    new double[]{x0[1] * cov[0][1]}, new double[]{1 - Math.pow(cov[0][1], 2)})).rand()[0];

            x0[1] = (new MultivariateGaussianDistribution(
                    new double[]{x0[0] * cov[0][1]}, new double[]{1 - Math.pow(cov[0][1], 2)})).rand()[0];

            // 收集样本集合
            if( k >= m ) {
                samples.add(Arrays.copyOf(x0, x0.length));
                sum_ += f(x0);
            }
        }

        return sum_ / n_samples;
    }

    public static void main(String[] args) {
        GibbsSamplingMethod GB = new GibbsSamplingMethod();
        int n_samples = 10000, m = 1000;
        ArrayList<double []> samples = new ArrayList<>();  // 随机样本列表
        long random_state = 0;
        double avg = GB.gibbs_sampling_method( new double[]{0, 0},
                new double [][]{{1, 0.5}, {0.5, 1}}, n_samples, m, random_state, samples);

        for( int i = 0; i < 5; i++ ) { //samples.size(); i++ ) {
            double [] sms = samples.get(i);
            printVectorElements(sms);
        }
        System.out.println("样本目标函数均值: " + avg);
        System.exit(0);
    }
}
