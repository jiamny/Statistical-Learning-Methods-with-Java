package com.github.jiamny.ml.Ch02_Linear_algebra;

import smile.math.Random;
import smile.plot.swing.Line;
import smile.plot.swing.LinePlot;
import smile.stat.distribution.BinomialDistribution;

import java.awt.*;

public class RandomNumbers {

    public static void main(String [] args) {
        Random rnd = new Random();
        rnd.setSeed(1);

        // next random number in in [0, 1)
        System.out.println("Next random number in [0, 1): " + rnd.nextDouble());

        // next random number in [0.3, 0.7]
        System.out.println("\nNext random number in [0.3, 0.7]: " + rnd.nextDouble(0.3, 0.7));

        // Binomial distribution
        int n = 10;
        BinomialDistribution bn = new BinomialDistribution(n, 0.15);

        double[][] data = new double[n][2];
        for (int i = 0; i < data.length; i++) {
            data[i][0] = i * 1.0;
            data[i][1] = bn.p(i);
        }

        // plot
        try {
            var plt =  LinePlot.of(data, Line.Style.SOLID, Color.BLUE);
            plt.canvas().setTitle("Binomial distribution (" + n + ", 0.2)");
            plt.canvas().setTitleColor(Color.RED);
            plt.canvas().window();
        } catch(Exception e ) {
            e.printStackTrace();
        }
    }
}
