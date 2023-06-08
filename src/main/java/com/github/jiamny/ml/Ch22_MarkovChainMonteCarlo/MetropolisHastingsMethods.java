package com.github.jiamny.ml.Ch22_MarkovChainMonteCarlo;

import smile.math.Random;
import smile.math.matrix.Matrix;
import smile.stat.distribution.MultivariateGaussianDistribution;

import java.util.ArrayList;
import java.util.Arrays;

import static com.github.jiamny.ml.utils.DataFrameHelper.range;
import static com.github.jiamny.ml.utils.StatisticHelper.printVectorElements;

public class MetropolisHastingsMethods {

    // 随机变量x=(x_1,x_2)的联合概率密度
    public double d1_pdf(double [] x) {
        if( 0 < x[0] &&  x[0] < x[1] )
            return x[0] * Math.pow(Math.E, -x[1]);
        else
            return 0;
    }

    //目标求均值函数
    public double f( double [] x) {
        return x[0] + x[1];
    }

    /*
    Metroplis-Hastings算法抽取样本
    :param x0: 初值（定义域中的任意一点即可）
    :param m: 收敛步数
    :param n: 迭代步数
    :return: 随机样本列表,随机样本的目标函数均值
     */
    public double metropolis_hastings_method(int m, int n, double [] x0,
                                             long random_state, ArrayList<double []> samples) {

        double sum_ = 0;                                // 目标求均值函数的和

        Random rd = new Random(random_state);

        int n_features = x0.length;

        Matrix dia = Matrix.diag(n_features, 1.0);

        // 循环执行n次迭代
        for(int k : range(n) ) {
            // 按照建议分布q(x,x')随机抽取一个候选状态
            // q(x,x')为均值为x，方差为1的正态分布
            // print('x0 ', x0, ' np.diag([1] * n_features) ', np.diag([1] * n_features))
            //x1 = np.random.multivariate_normal(x0, np.diag([1] * n_features), 1)[0]

            double [] x1 = (new MultivariateGaussianDistribution(x0, dia)).rand();

            // 计算接受概率
            double a = Math.min(1, d1_pdf(x1) / d1_pdf(x0));

            // 从区间(0,1)中按均匀分布随机抽取一个数u
            double u = rd.nextDouble();

            // 若u<=a，则转移状态；否则不转移
            if (u <= a)
                x0 = x1;

            // 收集样本集合
            if (k >= m) {
                samples.add(x0);
                sum_ += f(x0);
            }
        }
        return sum_ / (n - m);
    }

    /*
    单分量Metroplis-Hastings算法抽取样本

    :param x0: 初值（定义域中的任意一点即可）
    :param m: 收敛步数
    :param n: 迭代步数
    :param random_state: 随机种子
    :return: 随机样本列表,随机样本的目标函数均值
     */
    public double single_component_metropolis_hastings_method(int m, int n, double [] x0,
                                                              long random_state, ArrayList<double []> samples) {

        double sum_ = 0;                                // 目标求均值函数的和

        Random rd = new Random(random_state);
        int n_features = x0.length;
        Matrix dia = Matrix.diag(n_features, 1.0);

        int j = 0;   // 当前正在更新的分量

        // 循环执行n次迭代
        for(int  k : range(n) ) {
            // 按照建议分布q(x,x')随机抽取一个候选状态
            // q(x,x')为均值为x，方差为1的正态分布
            //x1 = x0.copy()
            double [] x1 = (double [])Arrays.copyOf(x0,x0.length);
            //np.random.multivariate_normal([x0[j]], np.diag([1]), 1)[0][0]
            x1[j] = (new MultivariateGaussianDistribution(
                    new double[]{x0[j]}, new double[]{1.0})).rand()[0];

            // 计算接受概率
            double a = Math.min(1, d1_pdf(x1) / d1_pdf(x0));

            // 从区间(0,1)中按均匀分布随机抽取一个数u
            double u = rd.nextDouble();

            // 若u<=a，则转移状态；否则不转移
            if( u <= a )
                x0 = x1;

            // 收集样本集合
            if( k >= m ) {
                samples.add(x0);
                sum_ += f(x0);
            }

            j = (j + 1) % n_features;
        }

        return sum_ / (n - m);
    }

    public static void main(String[] args) {
        MetropolisHastingsMethods MH = new MetropolisHastingsMethods();
        // -------------------------------------------------------------
        // Metroplis-Hastings算法抽取样本
        // -------------------------------------------------------------
        System.out.println("Metroplis-Hastings算法抽取样本: ");
        ArrayList<double []> samples = new ArrayList<>();  // 随机样本列表
        int m=1000, n=11000;
        double [] x0 = new double[]{5, 8};
        long random_state = 0;
        double avg = MH.metropolis_hastings_method(m, n, x0, random_state, samples);

        for( int i = 0; i < 5; i++ ) { //samples.size(); i++ ) {
            double [] sms = samples.get(i);
            printVectorElements(sms);
        }
        System.out.println("样本目标函数均值: " + avg);

        // -------------------------------------------------------------
        // 单分量Metroplis-Hastings算法抽取样本
        // -------------------------------------------------------------
        System.out.println("单分量Metroplis-Hastings算法抽取样本: ");
        samples = new ArrayList<>();  // 随机样本列表
        avg = MH.single_component_metropolis_hastings_method(m, n, x0, random_state, samples);

        for( int i = 0; i < 5; i++ ) { //samples.size(); i++ ) {
            double [] sms = samples.get(i);
            printVectorElements(sms);
        }
        System.out.println("样本目标函数均值: " + avg);

        System.exit(0);
    }
}
