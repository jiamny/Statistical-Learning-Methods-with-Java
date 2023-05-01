package com.github.jiamny.ml.Ch02_Linear_algebra;

import static com.github.jiamny.ml.utils.StatisticHelper.printVectorElements;
import static smile.base.cart.Loss.logistic;
import static smile.math.MathEx.*;

import smile.math.special.*;
import smile.math.matrix.*;

public class LinearAlgebraBasic {

    public static void main(String [] args) {
        System.out.println(System.getProperty("user.dir"));
        // Math Functions
        // logistic()
        int [] g = {5};
        System.out.println("logistic(5.0): " + logistic(g));

        // Special Functions
        System.out.println("Erf(1.0): " + Erf.erf(1.0));
        System.out.println(": " + Gamma.digamma(1.0));

        // Vector Operations
        double[] x = {1.0, 2.0, 3.0, 4.0};
        double[] y = {4.0, 3.0, 2.0, 1.0};
        System.out.println("norm(x): " + norm(x));
        System.out.println("unitize(y): ");
        unitize(y);
        printVectorElements(y);

        // Matrix Operations
        double[][] A = {
                {0.7220180, 0.07121225, 0.6881997},
                {-0.2648886, -0.89044952, 0.3700456},
                {-0.6391588, 0.44947578, 0.6240573}
        };
        double[][] B = {
                {0.6881997, -0.07121225, 0.7220180},
                {0.3700456, 0.89044952, -0.2648886},
                {0.6240573, -0.44947578, -0.6391588}
        };
        double[][] C = {
                {0.9527204, -0.2973347, 0.06257778},
                {-0.2808735, -0.9403636, -0.19190231},
                {0.1159052, 0.1652528, -0.97941688}
        };

        var a = Matrix.of(A);
        var b = Matrix.of(B);
        var c = Matrix.of(C);

        // matrix-vector operations
        double[] ax =  {1.0, 2.0, 3.0};
        double[] ay = {3.0, 2.0, 1.0};

        a.mv(1.0, ax, 1.5, ay); // y = a * xx + 1.5 *y

        System.out.println("y = a * xx + 1.5 *y:");
        printVectorElements(ay);

        // matrix-matrix operations
        a.add(b); // result saved in a, a.add(b) update a directly
        System.out.println("A + B:\n " + a);

        // Note that a * b are element-wise:
        a.mul(b); // result saved in a, a * b update a directly
        System.out.println("A * B:\n " + a);

        //  matrix multiplication
        a = Matrix.of(A);
        b = Matrix.of(B);
        c = Matrix.of(C);
        System.out.println("a.mm(b).mm(c):\n" + a.mm(b).mm(c) );

        a = Matrix.of(A);
        b = Matrix.of(B);
        c = Matrix.of(C);
        System.out.println("a'mm(b).mm(c):\n" + a.tm(b).mm(c) );

        //
        var ra = Matrix.randn( 300,  900);
        System.out.println("ra:\n" + ra );

        // Matrix Decomposition
        var kpa = a.clone();
        System.out.println("a.inverse():\n" + a.inverse());

        a = kpa.clone();
        var inv = a.inverse();
        System.out.println("inv:\n" + inv);

        a = kpa.clone();
        System.out.println("inv.mm(a):\n" + inv.mm(a));

        a = kpa.clone();
        var lu = a.lu();
        System.out.println("lu:\n" + lu.lu);
        System.out.println("ax:");
        printVectorElements(ax);

        var xs = lu.solve(ax);
        System.out.println("lu.solve(x):");
        printVectorElements(xs);

        System.exit(0);
    }
}
