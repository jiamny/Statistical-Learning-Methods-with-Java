package com.github.jiamny.ml.Ch24_PageRankAlgorithms;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import smile.math.matrix.fp32.Matrix;

import java.util.Arrays;

import static com.github.jiamny.ml.utils.DataFrameHelper.range;

public class PageRankAlgorithms {

    /*
    使用PageRank的基本定义求解PageRank值

    要求有向图是强联通且非周期性的

    :param M: 转移概率矩阵
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: PageRank值（平稳分布）
     */
    public NDArray pagerank_basic(NDArray M, double tol, int max_iter) {
        NDManager manager = NDManager.newBaseManager();
        int n_components = (int)M.getShape().getShape()[0];

        // 初始状态分布：均匀分布
        double [] pp = new double[n_components];
        Arrays.fill(pp, 1.0/n_components);
        NDArray pr0 = manager.create(pp);

        // 迭代寻找平稳状态
        for(int i :  range(max_iter) ) {
            NDArray pr1 = M.matMul(pr0);

            // 判断迭代更新量是否小于容差
            double ss = (pr0.sub(pr1).abs()).sum().toDoubleArray()[0];
            if( ss < tol ) {
                break;
            }
            pr0 = pr1;
        }
        return pr0;
    }

    /*
    PageRank的迭代算法

    :param M: 转移概率矩阵
    :param d: 阻尼因子
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: PageRank值（平稳分布）
     */
    public NDArray pagerank_1(NDArray M, double d, double tol, int max_iter) {
        NDManager manager = NDManager.newBaseManager();
        int n_components = (int)M.getShape().getShape()[0];

        // 初始状态分布：均匀分布
        double [] pp = new double[n_components];
        Arrays.fill(pp, 1.0/n_components);
        NDArray pr0 = manager.create(pp);

        // 迭代寻找平稳状态
        for(int i : range(max_iter) ) {
            //d * np.dot(M, pr0) + (1 - d) / n_components
            NDArray pr1 = ((M.matMul(pr0)).mul(d)).add((1.0 - d)/n_components);

            double ss = (pr0.sub(pr1).abs()).sum().toDoubleArray()[0];
            if( ss < tol ) {
                break;
            }
            pr0 = pr1;
        }

        return pr0;
    }

    /*
    计算一般PageRank的幂法

    :param M: 转移概率矩阵
    :param d: 阻尼因子
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: PageRank值（平稳分布）
     */
    public NDArray pagerank_2(NDArray M, double d, double tol, int max_iter) {
        NDManager manager = NDManager.newBaseManager();
        int n_components = (int)M.getShape().getShape()[0];


        // 选择初始向量x0：均匀分布
        double [] pp = new double[n_components];
        Arrays.fill(pp, 1.0/n_components);
        NDArray x0 = manager.create(pp);

        // 计算有向图的一般转移矩阵A
        NDArray A = (M.mul(d)).add((1 - d)/n_components);        //d * M + (1 - d) / n_components

        // 迭代并规范化结果向量
        for(int i : range(max_iter) ) {
            NDArray x1 = A.matMul(x0);
            NDArray xmx = x1.max();
            x1 = x1.div(xmx); // np.max(x1)

            double ss = (x0.sub(x1).abs()).sum().toDoubleArray()[0];
            if( ss < tol ) {
                break;
            }
            x0 = x1;
        }

        // 对结果进行规范化处理，使其表示概率分布
        NDArray x0s = x0.sum();
        x0 = x0.div(x0s);   //np.sum(x0)

        return x0;
    }

    /*
    PageRank的代数算法

    :param M: 转移概率矩阵
    :param d: 阻尼因子
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: PageRank值（平稳分布）
     */
    public NDArray pagerank_3(NDArray M, double d, double tol, int max_iter) {
        NDManager manager = NDManager.newBaseManager();
        int n_components = (int)M.getShape().getShape()[0];


        // 计算第一项：(I-dM)^-1
        NDArray dia = manager.eye(n_components);
        NDArray r1 = (dia.sub(M.mul(d))).inverse(); //np.linalg.inv(np.diag([1] * n_components) - d * M)

        // 计算第二项：(1-d)/n 1
        double [] pt = new double[n_components];
        Arrays.fill(pt, (1 - d)/n_components);
        NDArray r2 = manager.create(pt); //np.array([(1 - d) / n_components] * n_components)

        return r1.matMul(r2);
    }

    public static void main(String[] args) {
        NDManager manager = NDManager.newBaseManager();
        PageRankAlgorithms PRA = new PageRankAlgorithms();
        double tol=1e-8;
        int max_iter=1000;

        // --------------------------------------------------
        // 使用PageRank的基本定义求解PageRank值
        // --------------------------------------------------
        System.out.println("------- 使用PageRank的基本定义求解PageRank值 -------");
        int n_components = 4;
        double [][] pd = {{0., 1.0 / 2, 1.0, 0.},
                {1. / 3, 0., 0., 1. / 2},
                {1. / 3, 0., 0., 1. / 2},
                {1. / 3, 1. / 2, 0., 0.}};

        NDArray P = manager.create(pd);
        System.out.println(PRA.pagerank_basic(P, tol, max_iter));  // [0.33 0.22 0.22 0.22]

        double [][] pd2 = {{0., 1. / 2, 0., 0.},
                {1. / 3, 0., 0., 1. / 2},
                {1. / 3, 0., 0., 1. / 2},
                {1. / 3, 1. / 2, 0., 0.}};

        System.out.println(PRA.pagerank_basic(manager.create(pd2), tol, max_iter)); // [0. 0. 0. 0.]

        // --------------------------------------------------
        // PageRank的迭代算法
        // --------------------------------------------------
        System.out.println("------- PageRank的迭代算法 -------");
        double d = 0.8;
        double [][] pt11 = {{0., 1. / 2, 0., 0.},
                {1. / 3, 0., 0., 1. / 2},
                {1. / 3, 0., 0., 1. / 2},
                {1. / 3, 1. / 2, 0., 0.}};

        System.out.println(PRA.pagerank_1(manager.create(pt11), d, tol, max_iter));  // [0.1  0.13 0.13 0.13]

        double [][] pt12 = {{0., 1. / 2, 0., 0.},
                {1. / 3, 0., 0., 1. / 2},
                {1. / 3, 0., 1., 1./ 2},
                {1. / 3, 1. / 2, 0., 0.}};

        System.out.println(PRA.pagerank_1(manager.create(pt12), d, tol, max_iter));  // [0.1  0.13 0.64 0.13]

        double [][] pt13 = {{0., 0., 1.},
                {1. / 2, 0., 0.},
                {1. / 2, 1., 0.}};

        System.out.println(PRA.pagerank_1(manager.create(pt13), d, tol, max_iter));   // [0.38 0.22 0.4 ]

        // --------------------------------------------------
        // 计算一般PageRank的幂法
        // --------------------------------------------------
        System.out.println("------- 计算一般PageRank的幂法 -------");
        double [][] pt21 = {{0., 1. / 2, 0., 0.},
                            {1. / 3, 0., 0., 1. / 2},
                            {1. / 3, 0., 0., 1. / 2},
                            {1. / 3, 1. / 2, 0., 0.}};

        System.out.println(PRA.pagerank_2(manager.create(pt21), d, tol, max_iter)); // [0.2  0.27 0.27 0.27]

        double [][] pt22 = {{0., 1. / 2, 0., 0.},
                {1. / 3, 0., 0., 1. / 2},
                {1. / 3, 0., 1., 1. / 2},
                {1. / 3, 1. / 2, 0., 0.}};

        System.out.println(PRA.pagerank_2(manager.create(pt22), d, tol, max_iter));  // [0.1  0.13 0.64 0.13]

        double [][] pt23 = {{0., 0., 1.},
                {1. / 2, 0., 0.},
                {1. / 2, 1., 0.}};

        System.out.println(PRA.pagerank_2(manager.create(pt23), d, tol, max_iter));  // [0.38 0.22 0.4 ]

        // --------------------------------------------------
        // PageRank的代数算法
        // --------------------------------------------------
        System.out.println("------- PageRank的代数算法 -------");

        double [][] pt31 = {{0., 1. / 2, 0., 0.},
                {1. / 3, 0., 0., 1. / 2},
                {1. / 3, 0., 0., 1. / 2},
                {1. / 3, 1. / 2, 0., 0.}};

        System.out.println(PRA.pagerank_3(manager.create(pt31), d, tol, max_iter)); // [0.1  0.13 0.13 0.13]

        double [][] pt32 = {{0., 1. / 2, 0., 0.},
                {1. / 3, 0., 0., 1. / 2},
                {1. / 3, 0., 1., 1. / 2},
                {1. / 3, 1. / 2, 0., 0.}};

        System.out.println(PRA.pagerank_3(manager.create(pt32), d, tol, max_iter)); // [0.1  0.13 0.64 0.13]

        double [][] pt33 = {{0., 0., 1.},
                {1. / 2, 0., 0.},
                {1. / 2, 1., 0.}};

        System.out.println(PRA.pagerank_3(manager.create(pt33), d, tol, max_iter));   // [0.38 0.22 0.4 ]

        System.exit(0);
    }
}
