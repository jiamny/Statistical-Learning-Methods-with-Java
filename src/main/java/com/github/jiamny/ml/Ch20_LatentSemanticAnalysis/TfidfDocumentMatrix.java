package com.github.jiamny.ml.Ch20_LatentSemanticAnalysis;

import smile.math.matrix.Matrix;

import java.util.*;

import static com.github.jiamny.ml.utils.DataFrameHelper.range;
import static com.github.jiamny.ml.utils.StatisticHelper.printVectorObjects;

public class TfidfDocumentMatrix {

    /**
     * 依据TF-IDF构造的单词-文本矩阵
     *
     *     @param D: 文本集合
     *     @return: 依据TF-IDF的单词-文本矩阵
     */
    public Matrix get_word_document_matrix(ArrayList<String []> D) {
        int n_samples = D.size();

        // 构造所有文本出现的单词的集合
        //Comparator cmp = new
        SortedSet<String> W = new TreeSet<>();
        for( String [] d : D ) {
            //W |= set(d)
            for( String s : d )
                if( ! W.contains(s) )
                    W.add(s);
        }

        // 构造单词列表及单词下标映射
        HashMap<String, Integer> mapping = new HashMap<>();

        for( int i = 0; i < W.size(); i++ ) {
            //System.out.print(W.toArray()[i] + " ");
            mapping.put((String)W.toArray()[i], i);
        }
        //System.out.println();
        int n_features = W.size();

        // 计算：单词出现在文本中的频数/文本中出现的所有单词的频数之和
        double [][] X = new double[n_features][n_samples];
        for( int i = 0; i < n_features; i++ )
            Arrays.fill(X[i],0.0);

        for( int i = 0; i < D.size(); i++ ) {
            String [] d = D.get(i);
            for( String w : d ) {
                //X[mapping[w], i] +=1
                X[mapping.get(w)][i] += 1;
            }
            //X[:,i] /=len(d)
            for(int j = 0; j < n_features; j++)
                X[j][i] /= d.length;
        }

        // 计算：包含单词的文本数/文本集合D的全部文本数
        double [] df = new double[n_features]; //np.zeros(n_features)
        for( String [] d : D ) {
            for( String s : d )
                df[mapping.get(s)] += 1;
        }

        // 构造单词-文本矩阵
        for( int i : range(n_features) ) {
            for( int j : range(n_samples) )
                X[i][j] *= Math.log(n_samples / df[i]);
        }

        return Matrix.of(X);
    }

    public static void main(String[] args) {
        TfidfDocumentMatrix TDM = new TfidfDocumentMatrix();

        ArrayList<String[]> D = new ArrayList<>();
        D.add(new String[]{"guide", "investing", "market", "stock"});
        D.add(new String[]{"dummies", "investing"});
        D.add(new String[]{"book", "investing", "market", "stock"});
        D.add(new String[]{"book", "investing", "value"});
        D.add(new String[]{"investing", "value"});
        D.add(new String[]{"dads", "guide", "investing", "rich", "rich"});
        D.add(new String[]{"estate", "investing", "real"});
        D.add(new String[]{"dummies", "investing", "stock"});
        D.add(new String[]{"dads", "estate", "investing", "real", "rich"});
        /*
                {{"guide", "investing", "market", "stock"},
                {"dummies", "investing"},
                {"book", "investing", "market", "stock"},
                {"book", "investing", "value"},
                {"investing", "value"},
                {"dads", "guide", "investing", "rich", "rich"},
                {"estate", "investing", "real"},
                {"dummies", "investing", "stock"},
                {"dads", "estate", "investing", "real", "rich"}};
         */
        String [] d = D.get(0);
        printVectorObjects(d);

        System.out.println( TDM.get_word_document_matrix(D) );
    }
}
