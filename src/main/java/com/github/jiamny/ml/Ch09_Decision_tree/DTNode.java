package com.github.jiamny.ml.Ch09_Decision_tree;

import java.io.Serializable;

public class DTNode implements Serializable {
    //Mention the serialVersionUID explicitly in order to avoid getting errors while deserializing.
    public static final long serialVersionUID = 438L;
    boolean leaf;
    int label = -1;      // only defined if node is a leaf
    int attribute = -1;      // only defined if node is not a leaf
    double threshold;   // only defined if node is not a leaf
    DTNode left = null, right = null; //the left and right child of a particular node. (null if leaf)

    DTNode() {
        leaf = true;
        threshold = Double.MAX_VALUE;
    }
}

