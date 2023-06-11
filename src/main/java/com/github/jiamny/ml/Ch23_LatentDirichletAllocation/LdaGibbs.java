package com.github.jiamny.ml.Ch23_LatentDirichletAllocation;

import smile.math.Random;

import java.util.*;

import static com.github.jiamny.ml.utils.DataFrameHelper.range;
import static com.github.jiamny.ml.utils.StatisticHelper.*;

public class LdaGibbs {

    /*
    整理文本集合整理为数值化的文本的单词序列

    :param D: 文本集合
    :return: 数值化的文本的单词序列
     */
    public int check_words(ArrayList<String []> D,
                                                     ArrayList<ArrayList<Integer>> X) {
        int n_components = 0; // 单词集合中单词的个数
        HashMap<String, Integer> mapping = new HashMap<String, Integer>(); // 单词到数值化单词的映射

        for(String [] d : D) {
            ArrayList<Integer> x = new ArrayList<>();
            for( String word : d ) {
                if( ! mapping.containsKey(word) ){
                    mapping.put(word, n_components);
                    n_components += 1;
                }
                x.add(mapping.get(word));
            }
            X.add(x);
        }
        return n_components;
    }

    /*
    LDA吉布斯抽样算法
    :param D: 文本集合：每一行为1个文本，每一列为1个单词
    :param A: 超参数αlpha
    :param B: 超参数beta
    :param K: 话题个数
    :param n_iter: 迭代次数(直到结束燃烧期)
    :param random_state: 随机种子
    :return: 文本-话题计数矩阵(M×K矩阵),单词-话题计数矩阵(K×V矩阵),文本-话题概率矩阵(M×K矩阵),单词-话题概率矩阵(K×V矩阵)
     */
    public ArrayList<double [][]> lda_gibbs(ArrayList<String []> D, int K, int [] A,
                                            int [] B, int n_iter, long random_state) {

        ArrayList<ArrayList<Integer>> X = new ArrayList<>();
        int n_components = check_words(D, X);   // 数值化的文本的单词序列；单词数(V)
        int n_samples = D.size();               // 文本数(M)
        int [] n_features = new int[n_samples];
        //n_features = [len(X[m]) for m in range(n_samples)]
        // 文本中单词的个数(N_m)
        for( int m : range(n_samples) )
            n_features[m] = X.get(m).size();

        Random rd = new Random(random_state);
        // 初始化超参数alpha和beta：在没有其他先验知识的情况下，可以假设向量alpha和beta的所有分量均为1
        if( A == null ) {
            A = new int[K];
            Arrays.fill(A, 1);
        }

        if(B == null) {
            B = new int[n_components];
            Arrays.fill(B, 1);
        }
        int A_sum = (int)sum(A);
        int B_sum = (int)sum(B);

        // 初始化计数矩阵，设所有技术矩阵的元素的初值为0
        double [][] N_kv = new double[K][n_components];  // 单词-话题矩阵(K×V矩阵)
        for(int a : range(K))
            Arrays.fill(N_kv[a], 0.0);
        double [] N_k = new double[K];                   // 单词-话题矩阵的边缘计数
        Arrays.fill(N_k, 0.0);
        double [][] N_mk = new double[n_samples][K];     // 文本-话题矩阵(M×K矩阵)
        for(int a : range(n_samples))
            Arrays.fill(N_mk[a], 0.0);
        double [] N_m = new double[n_samples];           // 文本-话题矩阵的边缘计数
        Arrays.fill(N_m, 0.0);

        // 给文本的单词序列的每个位置上随机指派一个话题
        //Z = [[np.random.randint(0, K) for _ in range(n_features[m])] for m in range(n_samples)]
        ArrayList<ArrayList<Integer>>  Z = new ArrayList<ArrayList<Integer>>();
        for( int m : range(n_samples) ) {
            ArrayList<Integer> z = new ArrayList<>();
            for(int i : range(n_features[m]))
                z.add( rd.nextInt(K));
            Z.add(z);
        }

        // 根据随机指派的话题，更新计数矩阵
        for( int m : range(n_samples) ) {           // 遍历所有文本
            for(int n : range(n_features[m]) ) {    // 遍历第m个文本中的所有单词
                int v = (X.get(m)).get(n);          // 当前位置单词是第v个单词
                int k = (Z.get(m)).get(n);          // 当前位置话题是第k个话题
                N_kv[k][v] += 1;                    // 增加话题 - 单词计数
                N_k[k] += 1;                        // 增加话题 - 单词和计数
                N_mk[m][k] += 1;                    // 增加文本 - 话题计数
                N_m[m] += 1;                        // 增加文本 - 话题和计数
            }
        }

        // 循环执行以下操作，直到结束燃烧期
        for(int c : range(n_iter) ) {
            for(int m : range(n_samples) ) {            // 遍历所有文本
                for(int n : range(n_features[m]) ) {    // 遍历第m个文本中的所有单词
                    int v = (X.get(m)).get(n);          // 当前位置单词是第v个单词
                    int k = (Z.get(m)).get(n);          // 当前位置话题是第k个话题

                    // 对话题 - 单词矩阵和文本 - 话题矩阵中当期位置的已有话题的计数减1
                    N_kv[k][v] -= 1;                    // 减少话题 - 单词计数
                    N_k[k] -= 1;                        // 减少话题 - 单词和计数
                    N_mk[m][k] -= 1;                    // 减少文本 - 话题计数
                    N_m[m] -= 1;                        // 减少文本 - 话题和计数

                    // 按照满条件分布进行抽样（以下用于抽样的伪概率没有除以分母）
                    double [] p = new double[K];
                    Arrays.fill(p, 0.0);
                    for( int j : range(K) )
                        p[j] = ((N_kv[j][v] + B[v]) / (N_k[j] + B_sum)) * ((N_mk[m][j] + A[j]) / (N_m[m] + A_sum));

                    HashMap<Integer, Double> pp = new HashMap<>();
                    double ps = sum(p);
                    for( int j : range(K) ) {
                        p[j] /= ps;
                        pp.put(j, p[j]);
                    }
                    pp = sortByValueDesc(pp);

                    //k = np.random.choice(range(K), size = 1, p = p)[0]
                    k = -1;
                    double r = rd.nextDouble();
                    List<Map.Entry<Integer, Double> > list =
                            new LinkedList<Map.Entry<Integer, Double> >(pp.entrySet());
                    boolean first = true;
                    ps = 0.0;
                    for (Map.Entry<Integer, Double> aa : list) {
                        if( first ) {
                            if( r <= aa.getValue() ) {
                                k = aa.getKey();
                                break;
                            } else {
                                ps = aa.getValue();
                            }
                            first = false;
                        } else {
                            if( r > ps && r <= (ps + aa.getValue())) {
                                k = aa.getKey();
                                break;
                            }
                            ps += aa.getValue();
                        }
                    }

                    // 对话题 - 单词矩阵和文本 - 话题矩阵中当期位置的新话题的计数加1
                    N_kv[k][v] += 1;        //增加话题 - 单词计数
                    N_k[k] += 1;            //增加话题 - 单词和计数
                    N_mk[m][k] += 1;        //增加文本 - 话题计数
                    N_m[m] += 1;            //增加文本 - 话题和计数

                    //更新文本的话题序列
                    (Z.get(m)).set(n, k);
                }
            }
        }
        // 利用得到的样本计数，计算模型参数
        double [][] T = new double[n_samples][K];  // theta(M×K矩阵)
        for(int m : range(n_samples)) {
            for(int k : range(K))
                T[m][k] = N_mk[m][k] + A[k];
            double Ts = sum(T[m]);
            for(int k : range(K))
                T[m][k] /= Ts;
        }

        double [][] P = new double[K][n_components]; // phi(K×V矩阵)
        for(int k : range(K)) {
            for(int v : range(n_components))
                P[k][v] = N_kv[k][v] + B[v];
            double Ps = sum(P[k]);
            for(int v : range(n_components))
                P[k][v] /= Ps;
        }

        ArrayList<double [][]> res = new ArrayList<double [][]>();
        res.add(N_mk);
        res.add(N_kv);
        res.add(T);
        res.add(P);
        return res;
    }

    public static void main(String[] args) {
        LdaGibbs ldg = new LdaGibbs();
        ArrayList<String []>  example = new ArrayList<>();
        example.add( new String[]{"guide", "investing", "market", "stock"} );
        example.add( new String[]{"dummies", "investing"});
        example.add( new String[]{"book", "investing", "market", "stock"});
        example.add( new String[]{"book", "investing", "value"});
        example.add( new String[]{"investing", "value"});
        example.add( new String[]{"dads", "guide", "investing", "rich", "rich"});
        example.add( new String[]{"estate", "investing", "real"});
        example.add( new String[]{"dummies", "investing", "stock"});
        example.add( new String[]{"dads", "estate", "investing", "real", "rich"});

        int n_iter = 100;
        long random_state = 0;
        ArrayList<double [][]> res = ldg.lda_gibbs(example, 3,
                null, null, n_iter, random_state);

        System.out.println("文本-话题计数矩阵(M×K矩阵):");
        for(int i : range(res.get(0).length))
            printVectorElements(res.get(0)[i]);

        System.out.println("文本-话题概率矩阵(M×K矩阵):");
        for(int i : range(res.get(2).length))
            printVectorElements(res.get(2)[i], 2);

        System.out.println("单词-话题计数矩阵(K×V矩阵):");
        for(int i : range(res.get(1).length))
            printVectorElements(res.get(1)[i]);

        System.out.println("单词-话题概率矩阵(K×V矩阵):");
        for(int i : range(res.get(3).length))
            printVectorElements(res.get(3)[i], 2);

        System.exit(0);
    }
}
