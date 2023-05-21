package com.github.jiamny.ml.Ch05_Machine_learning_introduction;

import com.github.jiamny.ml.utils.MathExHelper;
import org.apache.commons.math3.analysis.polynomials.PolynomialFunction;
import org.apache.commons.math3.fitting.PolynomialCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoints;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.math.MathEx;
import smile.plot.swing.Legend;
import smile.plot.swing.Line;
import smile.plot.swing.LinePlot;
import smile.regression.LinearModel;
import smile.regression.OLS;
import smile.stat.distribution.GaussianDistribution;

import java.awt.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.function.DoubleFunction;
import java.util.stream.IntStream;

import static com.github.jiamny.ml.utils.StatisticHelper.linespace;
import static com.github.jiamny.ml.utils.StatisticHelper.printVectorElements;

public class Introduction {
    // ---------------------------------
    // 使用最小二乘法拟和曲线
    // ---------------------------------
    // 目标函数
    public double [] real_func(double [] x) {
        ArrayList<Double> sx = new ArrayList<>();
        for(int i = 0; i < x.length; i++)
            sx.add(Math.sin(2 * Math.PI * x[i]));
        //return np.sin(2 * np.pi * x)
        return sx.stream().mapToDouble(i->i).toArray();
    }

    public static void main(String [] args) {
        Introduction itd = new Introduction();

        // 十个点
        double [] x = linespace(0., 1., 30);
        //double [] x_points = linespace(0., 1., 1000);

        // 加上正态分布噪音的目标函数的值
        double [] y_ = itd.real_func(x);

        ArrayList<Double> sy = new ArrayList<>();
        var gaussian = new GaussianDistribution(0.0, 0.2);
        for( int i = 0; i < y_.length; i++ )
            sy.add( gaussian.rand() + y_[i]);
        //y = [np.random.normal(0, 0.1) + y1 for y1 in y_]
        double [] y = sy.stream().mapToDouble(i->i).toArray();

        //double [][] data = MathExHelper.arrayBindByCol(x, y);
        //String [] names = {"x", "y"};
        //DataFrame ols_dt = DataFrame.of(data, names);

        // Collect data.
        final WeightedObservedPoints obs = new WeightedObservedPoints();
        for( int i = 0; i < x.length; i++ )
            obs.add(x[i], y[i]);

        ArrayList<ArrayList<Double>> prd = new ArrayList<>();

        for( int degree = 1; degree <= 10; degree += 3 ) {
            System.out.println("M = " + degree);
            // Instantiate a third-degree polynomial fitter.
            final PolynomialCurveFitter fitter = PolynomialCurveFitter.create(degree);

            // Retrieve fitted parameters (coefficients of the polynomial function).
            final double[] coeff = fitter.fit(obs.toList());
            printVectorElements(coeff);
            PolynomialFunction approxFunction = new PolynomialFunction(coeff);

            ArrayList<Double> y_hat = new ArrayList<>();
            for( int i = 0; i < x.length; i++ ) {
                y_hat.add( approxFunction.value(x[i]) );
                //System.out.println(y[i] + " " + approxFunction.value(x[i]));
            }
            prd.add(y_hat);
        }

        // 可视化
        try {
            double[][] data = MathExHelper.arrayBindByCol(x, y_);
            Line dline = new Line(data, Line.Style.SOLID, 'o', Color.BLUE);
            double[][] dt1 = MathExHelper.arrayBindByCol(x, prd.get(0).stream().mapToDouble(i->i).toArray());
            Line x1line = new Line(dt1, Line.Style.DASH, '*', Color.RED);
            double[][] dt4 = MathExHelper.arrayBindByCol(x, prd.get(1).stream().mapToDouble(i->i).toArray());
            Line x4line = new Line(dt4, Line.Style.DOT_DASH, '+', Color.GREEN);
            double[][] dt7 = MathExHelper.arrayBindByCol(x, prd.get(2).stream().mapToDouble(i->i).toArray());
            Line x7line = new Line(dt7, Line.Style.LONG_DASH, 's', Color.CYAN);
            double[][] dt10 = MathExHelper.arrayBindByCol(x, prd.get(3).stream().mapToDouble(i->i).toArray());
            Line x10line = new Line(dt10, Line.Style.DOT, 'x', Color.MAGENTA);

            // M=1
            Line [] lines = {dline, x1line};
            Legend [] lgd = {new Legend("real", Color.BLUE),
                    new Legend("fitted curve", Color.RED)};
            var dlpt = new LinePlot(lines, lgd).canvas();
            dlpt.setLegendVisible(true);
            dlpt.setTitle("M = 1");
            dlpt.window();
            Thread.sleep(3000);
            dlpt.clear();

            // M=4
            Line [] lines2 = {dline, x4line};
            Legend [] lgd2 = {new Legend("real", Color.BLUE),
                    new Legend("fitted curve", Color.GREEN)};
            var dlpt2 = new LinePlot(lines2, lgd2).canvas();
            dlpt2.setLegendVisible(true);
            dlpt2.setTitle("M = 4");
            dlpt2.window();
            Thread.sleep(3000);
            dlpt2.clear();

            // M = 10
            Line [] lines3 = {dline, x10line};
            Legend [] lgd3 = {new Legend("real", Color.BLUE),
                    new Legend("fitted curve", Color.MAGENTA)};
            var dlpt3 = new LinePlot(lines3, lgd3).canvas();
            dlpt3.setLegendVisible(true);
            dlpt3.setTitle("M = 4");
            dlpt3.window();
            Thread.sleep(3000);
            dlpt3.clear();

        } catch(Exception e) {
            e.printStackTrace();
        }
    }
}
