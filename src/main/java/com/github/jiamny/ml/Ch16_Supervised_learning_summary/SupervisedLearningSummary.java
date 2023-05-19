package com.github.jiamny.ml.Ch16_Supervised_learning_summary;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import smile.plot.swing.Legend;
import smile.plot.swing.Line;
import smile.plot.swing.LinePlot;

import java.awt.*;

import static com.github.jiamny.ml.utils.StatisticHelper.linespace;

/*
监督学习方法总结

1 适用问题
监督学习可以认为是学习一个模型，使它能对给定的输入预测相应的输出。监督学习包括分类、标注、回归。
本篇主要考虑前两者的学习方法。分类问题是从实例的特征向量到类标记的预测问题；标注问题是从观测序列到标记序列
(或状态序列)的预测问题。可以认为分类问题是标注问题的特殊情况。 分类问题中可能的预测结果是二类或多类；
而标注问题中可能的预测结果是所有的标记序列，其数目是指数级的。感知机、𝑘近邻法、朴素贝叶斯法、决策树是简单的分类方法，
具有模型直观、方法简单、实现容易等特点；逻辑斯谛回归与最大熵模型、支持向量机、提升方法是更复杂但更有效的分类方法，
往往分类准确率更高；隐马尔可夫模型、条件随机场是主要的标注方法。通常条件随机场的标注准确率更事高。

2 模型
分类问题与标注问题的预测模型都可以认为是表示从输入空间到输出空间的映射.它们可以写成条件概率分布𝑃(𝑌|𝑋)
或决策函数𝑌=𝑓(𝑋)的形式。前者表示给定输入条件下输出的概率模型，后者表示输入到输出的非概率模型。
朴素贝叶斯法、隐马尔可夫模型是概率模型；感知机、𝑘近邻法、支持向量机、提升方法是非概率模型；而决策树、
逻辑斯谛回归与最大熵模型、条件随机场既可以看作是概率模型，又可以看作是非概率模型。直接学习条件概率分布𝑃(𝑌|𝑋)
或决策函数𝑌=𝑓(𝑋)的方法为判别方法，对应的模型是判别模型：感知机、𝑘近邻法、决策树、逻辑斯谛回归与最大熵模型、
支持向量机、提升方法、条件随机场是判别方法。首先学习联合概率分布𝑃(𝑋,𝑌)，从而求得条件概率分布𝑃(𝑌|𝑋)
的方法是生成方法，对应的模型是生成模型：朴素贝叶斯法、隐马尔可夫模型是生成方法。
决策树是定义在一般的特征空间上的，可以含有连续变量或离散变量。感知机、支持向量机、k近邻法的特征空间是欧氏空间(更一般地，
是希尔伯特空间)。提升方法的模型是弱分类器的线性组合，弱分类器的特征空间就是提升方法模型的特征空间。
感知机模型是线性模型；而逻辑斯谛回归与最大熵模型、条件随机场是对数线性模型；𝑘近邻法、决策树、支持向量机(包含核函数)、
提升方法使用的是非线性模型。

3 学习策略
在二类分类的监督学习中，支持向量机、逻辑斯谛回归与最大熵模型、提升方法各自使用合页损失函数、逻辑斯谛损失函数、指数损失函数

4 学习算法
统计学习的问题有了具体的形式以后，就变成了最优化问题。朴素贝叶斯法与隐马尔可夫模型的监督学习，最优解即极大似然估计值，
可以由概率计算公式直接计算。感知机、逻辑斯谛回归与最大熵模型、条件随机场的学习利用梯度下降法、
拟牛顿法等一般的无约束最优化问题的解法。 支持向量机学习，可以解凸二次规划的对偶问题。有序列最小最优化算法等方法。
决策树学习是基于启发式算法的典型例子。可以认为特征选择、生成、剪枝是启发式地进行正则化的极大似然估计。
提升方法利用学习的模型是加法模型、损失函数是指数损失函数的特点，启发式地从前向后逐步学习模型，以达到逼近优化目标函数的目的。
EM算法是一种迭代的求解含隐变量概率模型参数的方法，它的收敛性可以保证，但是不能保证收敛到全局最优。支持向量机学习、
逻辑斯谛回归与最大熵模型学习、条件随机场学习是凸优化问题，全局最优解保证存在。而其他学习问题则不是凸优化问题。
*/
public class SupervisedLearningSummary {

    public static void main(String [] args) {
        // 合页损失函数、逻辑斯谛损失函数、指数损失函数
        // 这3种损失函数都是0-1损失函数的上界，具有相似的形状。(见图)

        NDManager manager = NDManager.newBaseManager();
        //x = np.linspace(start=-1, stop=2, num=1001, dtype=np.float)
        double [] xdt = linespace(-1, 2, 1001);
        NDArray x = manager.create(xdt);
        // logi = np.log(1 + np.exp(-x)) / math.log(2)
        // boost = np.exp(-x)
        NDArray boost = x.mul(-1).exp();
        NDArray logi = (boost.add(1).log()).div(Math.log(2));

        NDArray y_01 = x.lt(0);             // < 0
        NDArray y_hinge = x.mul(-1).add(1); //1.0 - x
        y_hinge.set(y_hinge.lt(0), 0);      //y_hinge[y_hinge < 0] = 0
        System.out.println(y_01.getBoolean(0));
        System.out.println(y_hinge.getDouble(0));
        System.out.println(boost.getDouble(0));
        System.out.println(logi.getDouble(0));

        System.out.println(x.getShape().toString());
        double [][] data = new double[(int)x.getShape().getShape()[0]][2];
        System.out.println(data.length);

        try {
            for( int i = 0; i < data.length; i++) {
                data[i][0] = x.getDouble(i);
                data[i][1] = y_01.getBoolean(i) ? 1.0 : 0.0;
            }

            Line [] lines = new Line[4];
            lines[0] = new Line(data, Line.Style.SOLID, 'o', Color.GREEN);

            double [][] data2 = new double[(int)x.getShape().getShape()[0]][2];
            for( int i = 0; i < data.length; i++) {
                data2[i][0] = x.getDouble(i);
                data2[i][1] = y_hinge.getDouble(i);
            }
            lines[1] = new Line(data2, Line.Style.DASH, '*', Color.BLUE);

            double [][] data3 = new double[(int)x.getShape().getShape()[0]][2];
            for( int i = 0; i < data.length; i++) {
                data3[i][0] = x.getDouble(i);
                data3[i][1] = boost.getDouble(i);
            }
            lines[2] = new Line(data3, Line.Style.DOT_DASH, 'x', Color.MAGENTA);

            double [][] data4 = new double[(int)x.getShape().getShape()[0]][2];
            for( int i = 0; i < data.length; i++) {
                data4[i][0] = x.getDouble(i);
                data4[i][1] = logi.getDouble(i);
            }
            lines[3] = new Line(data4, Line.Style.LONG_DASH, '+', Color.RED);
            Legend[] legends = {
                    new Legend("0/1 Loss", Color.GREEN),
                    new Legend("Hinge Loss", Color.BLUE),
                    new Legend("Boost Loss", Color.MAGENTA),
                    new Legend("Logistic Loss", Color.RED),
            };
            var dlpt = new LinePlot(lines, legends).canvas();
            dlpt.setLegendVisible(true);
            dlpt.window();
            Thread.sleep(3000);
            dlpt.clear();
        } catch(Exception e) {
            e.printStackTrace();
        }
    }
}
