package com.github.jiamny.ml.Ch17_Clustering;

import com.github.jiamny.ml.utils.DataFrameHelper;
import com.github.jiamny.ml.utils.MathExHelper;
import com.github.jiamny.ml.utils.ThreeTuple;
import smile.data.*;
import smile.math.MathEx;
import smile.plot.swing.Line;
import smile.plot.swing.LinePlot;

import java.awt.*;
import java.util.*;
import java.util.List;

import static com.github.jiamny.ml.utils.DataFrameHelper.*;
import static com.github.jiamny.ml.utils.StatisticHelper.combGenerate;
import static com.github.jiamny.ml.utils.StatisticHelper.minIndex;

public class KmeansClustering {

    // 定义标准化函数，对每一列特征进行min-max标准化，将数据缩放到0-1之间
    // 标准化处理对于计算距离的机器学习方法是非常重要的，因为特征的尺度不同会导致计算出来的距离倾向于尺度大的特征，
    // 为保证距离对每一列特征都是公平的，必须将所有特征缩放到同一尺度范围内
    /*
    INPUT:
    Xarray - (array) 特征数据数组

    OUTPUT:
    Xarray - (array) 标准化处理后的特征数据数组
     */
    public DataFrame Normalize(DataFrame Xarray) {
        String [] names = Xarray.names();
        double [][] data = Xarray.toMatrix().toArray();
        for(int f : range(Xarray.ncol()) ) {
            double maxf = MathEx.max(Xarray.select(f).toMatrix().toArray());
            double minf = MathEx.min(Xarray.select(f).toMatrix().toArray());

            for(int n : range(Xarray.nrow()) )
                data[n][f] = (data[n][f] - minf) / (maxf - minf);
        }
        return DataFrame.of(data, names);
    }

    /*
    INPUT:
    Xi - (array) 第i条特征数据
    Xj - (array) 第j条特征数据

    OUTPUT:
    dist - (float) 两条数据的欧式距离
     */
    public double cal_distance(double [] xi, double [] xj) {
        double dist = 0;
        for(int col : range(xi.length) )
            dist += Math.pow((xi[col]-xj[col]), 2);
        dist = Math.sqrt(dist);
        return dist;
    }

    // 定义计算类中心的函数，以当前类中所包含数据的各个特征均值作为新的新的类中心
    /*
    INPUT:
    group - (list) 类所包含的数据列表
    Xarray - (array) 特征数据数组

    OUTPUT:
    center - (array) 新的类中心
     */

    public DataFrame cal_groupcenter(ArrayList<Integer> group, DataFrame Xarray) {
        //center = np.zeros(Xarray.shape[1])
        double [][] center = new double[1][Xarray.ncol()];

        for(int i : range(Xarray.ncol())) {
            for(int n : group.stream().mapToInt(x -> x).toArray())
                center[0][i] += Xarray.getDouble(n, i);  // 计算当前类中第i个特征的数据之和
        }
        for(int i : range(Xarray.ncol()))
            center[0][i] = center[0][i] / Xarray.nrow();  //计算各个特征的均值

        return DataFrame.of(center);
    }

    // 定义计算调整兰德系数(ARI)的函数，调整兰德系数是一种聚类方法的常用评估方法
    /*
    INPUT:
    group_dict - (dict) 类别字典
    Ylist - (list) 类别标签列表
    k - (int) 设定的类别数

    OUTPUT:
    (int) 调整兰德系数
     */

    public double Adjusted_Rand_Index(HashMap<Integer, ArrayList<Integer>> group_dict,
                               DataFrame Ylist, int k) {
        // 定义一个数组，用来保存聚类所产生的类别标签与给定的外部标签各类别之间共同包含的数据数量
        int [][] group_array = new int[k][k];
        for( int i = 0; i < k; i++ )
            Arrays.fill(group_array[i], 0);

        // Ylist保存的标签为字符串，用ylabel来保存各个标签，在y_dict中类别以标签在ylabel列表中的索引值来表示类
        ArrayList<String> labs = new ArrayList<>();
        for(int n : range(Ylist.nrow()))
            labs.add(Ylist.getString(n, 0));
        Object [] L = unique(labs.toArray());
        HashMap<String, Integer> ylabel = new HashMap<>();
        for( int i : range(L.length))
            ylabel.put(L[i].toString(), i);


        //定义一个空字典，用来保存外部标签中各类所包含的数据，结构与group_dict相同
        HashMap<Integer, ArrayList<Integer>> y_dict = new HashMap<>();
        for(int i : range(k) )
            y_dict.put(i, new ArrayList<Integer>());

        for( int i : range(Ylist.nrow()) ) {
            y_dict.get(ylabel.get(Ylist.getString(i, 0))).add(i);
        }

        // 循环计算group_array的值
        for(int i : range(k)) {
            for(int j : range(k) ) {
                for(int n : range(Ylist.nrow()) ) {
                    //如果数据n同时在group_dict的类别i和y_dict的类别j中，group_array[i][j] 的数值加一
                    if (group_dict.get(i).contains(n) && y_dict.get(j).contains(n))
                        group_array[i][j] += 1;
                }
            }
        }

        double RI = 0;                //定义兰德系数(RI)
        int [] sum_i = new int[k];    //定义一个数组，用于保存聚类结果group_dict中每一类的个数
        Arrays.fill(sum_i, 0);
        int [] sum_j = new int[k];    //定义一个数组，用于保存外部标签y_dict中每一类的个数
        Arrays.fill(sum_j, 0);

        for(int i : range(k) ) {
            for(int j : range(k) ) {
                sum_i[i] += group_array[i][j];
                sum_j[j] += group_array[i][j];
                //combGenerate用于计算group_array[i][j] 中两两组合的组合数
                if( group_array[i][j] >= 2 ) {
                    RI += combGenerate(group_array[i][j], 2).size();
                }
            }
        }

        double ci = 0.0;  //ci保存聚类结果中同一类中的两两组合数之和
        double cj = 0.0;  //cj保存外部标签中同一类中的两两组合数之和
        for(int i : range(k)) {
            if(sum_i[i] >= 2)
                ci += combGenerate(sum_i[i], 2).size();
        }
        for(int j : range(k)) {
            if(sum_j[j] >= 2)
                cj += combGenerate(sum_j[j], 2).size();
        }
        double E_RI = (ci * cj) / (combGenerate(Ylist.nrow(), 2).size()); //计算RI的期望
        double max_RI = (ci + cj) / 2;                                  //计算RI的最大值
        return ((RI - E_RI) / (max_RI-E_RI));   //返回调整兰德系数的值
    }

    // 定义k均值聚类函数
    /*
    INPUT:
    Xarray - (array) 特征数据数组
    k - (int) 设定的类别数
    iters - (int) 设定的迭代次数

    OUTPUT:
    group_dict - (dict) 类别字典
    scores - (int) 每次迭代的ARI得分列表
     */

    public ThreeTuple<HashMap<Integer, ArrayList<Integer>>, ArrayList<Double>, DataFrame> Kmeans(
            DataFrame Xarray, DataFrame Ylist, int k, int iters) {
        Random random = new Random(234);
        int [] center_inds = new int[k];  //从特征数据中随机抽取k个数据索引
        for(int i : range(k) )
            center_inds[i] = random.nextInt(Xarray.nrow());

        //将这k个数据索引所对应的特征数据作为初始的k个聚类中心
        DataFrame centers = Xarray.of(center_inds);     //[Xarray[ci] for ci in center_inds]
        //定义一个空列表用来保存每次迭代的ARI得分
        ArrayList<Double> scores = new ArrayList<>();
        HashMap<Integer, ArrayList<Integer>> group_dict = null;

        for(int i : range(iters) ) {
            // 定义一个空字典，用于保存聚类所产生的所有类别，其中字典的键为类别标签，
            // 值为类别所包含的数据列表，以索引表示每条数据
            //group_dict = {i:[]for i in range(k)}
            group_dict = new HashMap<>();
            for(int j : range(k) )
                group_dict.put(j, new ArrayList<Integer>());

            System.out.printf("%d/%d\n", i + 1, iters);
            // 循环计算每条数据到各个聚类中心的距离
            for(int n : range(Xarray.nrow())) {
                double [] dists = new double[k];  //保存第n条数据到各个聚类中心的距离
                for(int ci : range(k)) {
                    double dist = cal_distance(Xarray.of(n).toMatrix().toArray()[0],
                            centers.of(ci).toMatrix().toArray()[0]);
                    dists[ci] = dist;
                }
                //取距离最近的中心所在的类
                //System.out.println(Arrays.toString(dists));
                int g = minIndex(dists);
                //System.out.println("g: " + g);
                group_dict.get(g).add(n);           //将第n条数据的索引n保存到g类
            }
            //print(group_dict)
            double [][] c_data = centers.toMatrix().toArray();
            for(int j : range(k) ) {
                //根据每一类所包含的数据重新计算类中心
                DataFrame new_ct = cal_groupcenter(group_dict.get(j), Xarray);
                for( int m : range(new_ct.ncol()))
                    c_data[j][m] = new_ct.getDouble(0, m);
            }
            centers = DataFrame.of(c_data);

            scores.add(Adjusted_Rand_Index(group_dict, Ylist, k));  //将该轮迭代的ARI得分保存到scores列表
        }

        return new ThreeTuple<>(group_dict, scores, centers);
    }

    public static void main(String[] args) {
        Long start = System.currentTimeMillis();

        KmeansClustering KMC = new KmeansClustering();
        HashMap<Integer, ArrayList<Integer>> y_dict = new HashMap<>();
        for(int i : range(3) )
            y_dict.put(i, new ArrayList<Integer>());

        y_dict.get(0).add(1);
        y_dict.get(0).add(2);
        y_dict.get(0).add(3);

        System.out.println(y_dict.get(0).contains(2));

        List<int[]> comb = combGenerate(50, 2);
        System.out.println(comb.size());

        try {
            String fname = "./data/iris.data";

            DataFrame data = DataFrameHelper.loadIrisData(fname);

            DataFrame X = data.select(new int[]{0, 1, 2, 3});
            DataFrame y = data.select(5);

            System.out.println(X.summary());
            System.out.println(y);
            X = KMC.Normalize(X);  //对特征数据进行标准化处理

            int k = 3;          //设定聚类数为3
            int iters = 2;      //设定迭代次数为10

            var res = KMC.Kmeans(X, y, k, iters);
            DataFrame centers = res.third;
            System.out.println(centers);

            double[][] dt = MathExHelper.arrayBindByCol(Arrays.stream(range(iters)).asDoubleStream().toArray(),
                    res.second.stream().mapToDouble(i->i).toArray());

            Line dline = new Line(dt, Line.Style.SOLID, '.', Color.BLUE);
            var dlpt = new LinePlot(dline).canvas();
            dlpt.window();
            Thread.sleep(3000);
            dlpt.clear();
        } catch(Exception e) {
            e.printStackTrace();
        }
        System.out.printf("Time span: %dms\n",  (System.currentTimeMillis() - start));
        //System.exit(0);
    }
}
