package com.github.jiamny.ml.Ch17_Clustering;

import com.github.jiamny.ml.utils.DataFrameHelper;
import smile.data.DataFrame;
import smile.math.matrix.Matrix;
import smile.plot.swing.Canvas;
import smile.plot.swing.Plot;
import smile.plot.swing.ScatterPlot;

import javax.swing.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import static com.github.jiamny.ml.utils.DataFrameHelper.euler_distance;
import static com.github.jiamny.ml.utils.DataFrameHelper.range;
import static com.github.jiamny.ml.utils.StatisticHelper.printVectorElements;


/*
层次聚类
    聚合（自下而上）：聚合法开始将每个样本各自分裂到一个类，之后将相距最近的两类合并，建立一个新的类，
    重复次操作知道满足停止条件，得到层次化的类别。

    分裂（自上而下）： 分裂法开始将所有样本分到一个类，之后将已有类中相距最远的样本分到两个新的类，
    重复此操作直到满足停止条件，得到层次化的类别。

k均值聚类
    k均值聚类是基于中心的聚类方法，通过迭代，将样本分到k个类中，使得每个样本与其所属类的中心或均值最近，
    得到k个平坦的，非层次化的类别，构成对空间的划分。
 */

// 定义聚类数的节点

class ClusterNode {
    public ClusterNode left = null, right = null;
    public int id, count;
    public double distance;
    public DataFrame vec;
    /**
     * @param vec: 保存两个数据聚类后形成新的中心
     * @param left: 左节点
     * @param right:  右节点
     * @param distance: 两个节点的距离
     * @param id: 用来标记哪些节点是计算过的
     * @param count: 这个节点的叶子节点个数
     */
    public ClusterNode(DataFrame vec, ClusterNode left, ClusterNode right,
                       double distance, int id, int count) {
        this.vec = vec;
        this.left = left;
        this.right = right;
        this.distance = distance;
        this.id = id;
        this.count = count;
    }
}

// 层次聚类（聚合法）
class Hierarchical {
    public int k = 0;
    public ArrayList<Integer> labels;
    public ArrayList<ClusterNode> nodes = null;

    public Hierarchical(int k) {
        this.k = k;
        labels = new ArrayList<Integer>();
    }

    public void fit(DataFrame x) {
        ArrayList<ClusterNode> nodes = new ArrayList<ClusterNode>();
        for( int i = 0; i < x.nrow(); i++ ) {
            nodes.add(new ClusterNode(x.of(i), null, null, -1, i, 1));
        }

        HashMap<String, Double> distances = new HashMap<>();
        int point_num = x.nrow(), feature_num = x.ncol();
        for( int i = 0;  i < point_num; i++ )
            labels.add(-1);

        int currentclustid = -1;
        while( nodes.size() > k ) {
            double min_dist = Double.POSITIVE_INFINITY;
            int nodes_len = nodes.size();
            ArrayList<Integer> closest_part = new ArrayList<>();

            for( int i : range(nodes_len - 1) ){
                System.out.println("i + 1: " + (i + 1) + " nodes_len " + nodes_len);
                for( int j : range(i + 1, nodes_len) ) {
                    String d_key = nodes.get(i).id + "_" + nodes.get(j).id;
                    //System.out.println("d_key " + d_key);
                    if( ! distances.containsKey(d_key) ) {
                        distances.put(d_key,
                                euler_distance(nodes.get(i).vec.toMatrix(),
                                nodes.get(j).vec.toMatrix()));
                    }
                    double d = distances.get(d_key);
                    //System.out.println("i: " + i + " j: " + j);
                    if( d < min_dist ) {
                        min_dist = d;
                        closest_part.add(i);
                        closest_part.add(j);    //(i, j)
                    }
                }
            }

            int part1 = closest_part.get(0), part2 = closest_part.get(1);
            ClusterNode node1 = nodes.get(part1), node2 = nodes.get(part2);

            double [][] new_vec = new double[1][feature_num];
            for(int i : range(feature_num) )
                new_vec[0][i] = (node1.vec.getDouble(0, i) * node1.count +
                        node2.vec.getDouble(0, i) * node2.count ) / (node1.count + node2.count);

            ClusterNode new_node = new ClusterNode(DataFrame.of(new_vec),
                    node1,
                    node2,
                    min_dist,
                    currentclustid,
                    node1.count + node2.count);
            currentclustid -= 1;
            //del nodes[part2], nodes[part1]
            nodes.remove(part2);
            nodes.remove(part1);

            nodes.add(new_node);
        }
        this.nodes = nodes;
        calc_label();
    }

    public void calc_label() {
        // 调取聚类的结果
        for( int i = 0; i < nodes.size(); i++ ) {
            // 将节点的所有叶子节点都分类
            leaf_traversal(nodes.get(i), i);
        }
    }

    public void leaf_traversal( ClusterNode node , int label ) {
        // 递归遍历叶子节点
        if( node.left == null && node.right == null )
            labels.set(node.id, label);
        if( node.left != null )
            leaf_traversal(node.left, label);
        if( node.right != null )
            leaf_traversal(node.right, label);
    }
}

public class Clustering {

    public static void main(String [] args) {
        try {
            String fname = "./data/iris.data";

            DataFrame data = DataFrameHelper.loadIrisData(fname);
            Canvas canvas = ScatterPlot.of(data, "sepallength", "sepalwidth", "class", '*').canvas();
            canvas.setAxisLabels("sepallength", "sepalwidth");
            canvas.setTitle("Iris data");
            JFrame frame = canvas.window();
            frame.setVisible(true);

            System.out.println(data.select("class"));

            //int [] data_idx = new int[data.ncol()-1];
            //for(int i = 0; i < (data.ncol()-3); i++ )
            //    data_idx[i] = i;
            DataFrame X = data.select("sepallength", "sepalwidth");
            DataFrame y = data.select("class");

            System.out.println(X.summary());
            /*
            double [][] p1 = {{5}, {6}, {7}, {8}};
            double [][] p2 = {{1}, {2}, {3}, {4}};
            Matrix pt1 = Matrix.of(p1);
            Matrix pt2 = Matrix.of(p2);

            Matrix d = pt1.sub(pt2);
            double distance = d.transpose().mm(d).get(0, 0);
            System.out.println(d.transpose().mm(d));
            System.out.println("Math.sqrt(distance) " + Math.sqrt(distance));
             */

            Hierarchical mH = new Hierarchical(3);
            mH.fit(X);
            int [] labels = mH.labels.stream().mapToInt(i -> i).toArray();
            printVectorElements(labels);

            // visualize result
            int [][] plabels = new int[labels.length][1];
            for( int i : range(labels.length) )
                plabels[i][0] = labels[i];

            DataFrame Ldata = DataFrame.of(plabels, "labels"); // new String[]{"labels"}
            X = X.merge(Ldata);
            canvas = ScatterPlot.of(X, "sepallength", "sepalwidth", "labels", 'x').canvas();
            canvas.setAxisLabels("sepallength", "sepalwidth");
            canvas.setTitle("Hierarchical clustering with k=3");
            frame = canvas.window();
            frame.setVisible(true);
            Thread.sleep(3000);

        } catch(Exception e) {
            e.printStackTrace();
        }
    }
}
