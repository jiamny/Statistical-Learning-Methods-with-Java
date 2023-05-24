package com.github.jiamny.ml.Ch15_ConditionalRandomField;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static com.github.jiamny.ml.utils.DataFrameHelper.range;
import static com.github.jiamny.ml.utils.StatisticHelper.printVectorElements;

public class ViterbiAlgorithm {
    public int t1(int y0, int y1, int i) {
        Set<Integer> t = new HashSet<>();
        t.add(1);
        t.add(2);
        return (y0 == 0 && y1 == 1 && t.contains(i)) ? 1 : 0;
    }

    public int t2(int y0, int y1, int i) {
        Set<Integer> t = new HashSet<>();
        t.add(1);
        return (y0 == 0 && y1 == 0 && t.contains(i)) ? 1 : 0;
    }

    public int t3(int y0, int y1, int i) {
        Set<Integer> t = new HashSet<>();
        t.add(2);
        return (y0 == 1 && y1 == 0 && t.contains(i)) ? 1 : 0;
    }

    public int t4(int y0, int y1, int i) {
        Set<Integer> t = new HashSet<>();
        t.add(1);
        return (y0 == 1 && y1 == 0 && t.contains(i)) ? 1 : 0;
    }
    public int t5(int y0, int y1, int i) {
        //return int(y0 == 1 and y1 == 1 and i in {2})
        Set<Integer> t = new HashSet<>();
        t.add(2);
        return (y0 == 1 && y1 == 1 && t.contains(i)) ? 1 : 0;
    }

    public int s1(int y0, int i) {
        //return int(y0 == 0 and i in {0})
        Set<Integer> t = new HashSet<>();
        t.add(0);
        return (y0 == 0 && t.contains(i)) ? 1 : 0;
    }

    public int  s2(int y0, int i) {
        //return int(y0 == 1 and i in {0, 1})
        Set<Integer> t = new HashSet<>();
        t.add(0);
        t.add(1);
        return (y0 == 1 && t.contains(i)) ? 1 : 0;
    }

    public int s3(int y0, int i) {
        //return int(y0 == 0 and i in {1, 2})
        Set<Integer> t = new HashSet<>();
        t.add(1);
        t.add(2);
        return (y0 == 0 && t.contains(i)) ? 1 : 0;
    }

    public int s4(int y0, int i) {
        //return int(y0 == 1 and i in {2})
        Set<Integer> t = new HashSet<>();
        t.add(2);
        return (y0 == 1 && t.contains(i)) ? 1 : 0;
    }

    /**
     * 维特比算法预测状态序列
     *
     *     @param w1: 模型的转移特征权重
     *     @param w2: 模型的状态特征权重
     *     @param x: 需要计算的观测序列
     *     @param n_state: 状态的可能取值数
     *     @return: 最优可能的状态序列
     */
    public int [] viterbi_algorithm(double [] w1, double [] w2, int [] x, int n_state) {
        int n_transfer_features = 5;    // 转移特征数
        int n_state_features = 4;       // 状态特征数
        int n_position = x.length;      // 序列中的位置数

        //定义状态矩阵
        double [][] dp = new double[n_position][n_state];   // 概率最大值
        for( int i : range(n_position))
            Arrays.fill(dp[i], 0.0);

        //[[-1] * n_state for _ in range(n_position)]
        int [][]    last = new int[n_position][n_state];   // 上一个结点
        for( int i : range(n_position))
            Arrays.fill(last[i], -1);

        // 处理t=0的情况
        for(int i : range(n_state)) {
            for(int l : range(n_state_features) ) {
                //dp[0][i] += w2[l] * state_features[l] (y0 = i, x = x, i = 0)
                switch (l) {
                    case 0 -> dp[0][i] += w2[l] * s1(i, 0);
                    case 1 -> dp[0][i] += w2[l] * s2(i, 0);
                    case 2 -> dp[0][i] += w2[l] * s3(i, 0);
                    case 3 -> dp[0][i] += w2[l] * s4(i, 0);
                    default -> {
                    }
                }
            }
        }

        // 处理t>0的情况
        for(int t : range(1, n_position) ) {
            for(int i : range(n_state)) {
                for(int j : range(n_state)) {
                    double d = dp[t - 1][i];
                    for(int k : range(n_transfer_features)) {
                        //d += w1[k] * transfer_features[k] (y0 = i, y1 = j, x = x, i = t)
                        switch (k) {
                            case 0 -> d += w1[k] * t1(i, j, t);
                            case 1 -> d += w1[k] * t2(i, j, t);
                            case 2 -> d += w1[k] * t3(i, j, t);
                            case 3 -> d += w1[k] * t4(i, j, t);
                            case 4 -> d += w1[k] * t5(i, j, t);
                            default -> {
                            }
                        }
                    }
                    for(int l : range(n_state_features)) {
                        //d += w2[l] * state_features[l] (y0 = j, x = x, i = t)
                        switch (l) {
                            case 0 -> d += w2[l] * s1(j, t);
                            case 1 -> d += w2[l] * s2(j, t);
                            case 2 -> d += w2[l] * s3(j, t);
                            case 3 -> d += w2[l] * s4(j, t);
                            default -> {
                            }
                        }
                    }
                    //print((i, j), "=", d)
                    if(d >= dp[t][j]) {
                        dp[t][j] = d;
                        last[t][j] = i;
                    }
                }
            }
        }
        // 计算最优路径的终点
        int best_end = 0;
        double best_gamma = 0;
        for(int i : range(n_state)) {
            if( dp[dp.length -1][i] > best_gamma ) {
                best_end = i;
                best_gamma = dp[dp.length -1][i];
            }
        }

        //计算最优路径
        int [] ans = new int[n_position]; //[0] * (n_position - 1) + [best_end]
        ans[n_position-1] = best_end;
        for(int t = n_position - 1; t > 0; t--)
            ans[t - 1] = last[t][ans[t]];
        //for t in range(n_position - 1, 0, -1):
        //ans[t - 1] = last[t][ans[t]]
        return ans;
    }

    public static void main(String[] args) {

        ViterbiAlgorithm vb = new ViterbiAlgorithm();
        double [] w1 = {1, 0.6, 1, 1, 0.2};
        double [] w2 = {1, 0.5, 0.8, 0.5};
        int [] x = {1, 0, 1};

        int [] ans = vb.viterbi_algorithm(w1, w2, x, 2);
        printVectorElements(ans);
    }
}
