package com.github.jiamny.ml.Ch10_LogisticRegressionAndMaximumEntropy;

import smile.data.DataFrame;

public class LogisticRegression {

    /**
     逻辑斯蒂回归训练过程
     @param trainDataList:训练集
     @param trainLabelList: 标签集
     @param iter: 迭代次数
     @return: 习得的w
     */
    public DataFrame logisticRegression( DataFrame trainDataList, DataFrame trainLabelList, int iter ) {
        if( iter < 200 )
            iter = 200;

        // 按照书本“6.1.2 二项逻辑斯蒂回归模型”中式6.5的规则，将w与b合在一起，
        // 此时x也需要添加一维，数值为1
        // 循环遍历每一个样本，并在其最后添加一个1
        int [][] append = new int[trainDataList.nrow()][1];
        for( int k = 0; k < append.length; k++ )
            append[k][0] = 1;
        trainDataList = trainDataList.merge(DataFrame.of(append));
    }
}
