package com.github.jiamny.ml.Ch01_Statistics;

import smile.math.special.Erf;
import smile.plot.swing.*;

import java.awt.*;
import java.lang.String;
import java.lang.Math;
import java.math.BigInteger;
import java.util.ArrayList;

import com.github.jiamny.ml.utils.*;
import static smile.math.special.Gamma.gamma;

public class Distribution {
    /**
     * @param x random variable
     * @param p possibility
     * @return random value
     */
    public double F(double x, double p) {
        if( x < 0.0 )
            return 0.0;
        else {
            if(x > 1.0)
                return 1.0;
            else
                return (1.0 - p);
        }
    }

    /**
     * @param x random value
     * @param n random value
     * @return random value
     */
    public double DF(double x, double n) {

        if( x < 1.0 )
            return 0.0;
        else {
            if( x > n )
                return 1.0;
            else
                return Math.floor(x) / n;
        }
    }

    public double CF(double x, double a, double b) {
        if( x < a) {
            return 0.0;
        } else {
            if( x > b ) {
                return 1.0;
            } else {
                return (x - a)/(b - a);
            }
        }
    }

    // Compute binomial coefficient
    public BigInteger binom(int n, int k) {
        BigInteger comb = BigInteger.valueOf(1);
        int x = Math.min(k, n - k);
        for(int i = 0; i < x; i++ )
            comb = comb.multiply( BigInteger.valueOf(((n - i) / (i + 1)) ));
        return comb;
    }

    public double BF(double x, double n, ArrayList<Double> cmf) {
        if(x < 0.0)
            return 0.0;
        else {
            if( x > n )
                return 1.0;
            else
                return cmf.get((int) x );
        }
    }

    public double phi(double x, double mu, double sigma) {
        return (1.0 + Erf.erf((x - mu) / (sigma * Math.sqrt(2.)))) / 2.0;
    }


    public static void main(String [] args)  {
        System.out.println(System.getProperty("user.dir"));
        Distribution demo = new Distribution();

        try {

            // --------------------------------
            // Bernoulli
            // --------------------------------
            double p = 0.3;
            double[][] dt = {{0.0, 1.0}, {1.0 - p, p}};
            Bar bar = new Bar(dt, 0.01, Color.BLUE);
            var plt = new BarPlot(bar).canvas();
            plt.setTitle("Bernoulli");
            plt.setAxisLabels("x", "p.m.f.");
            plt.window();
            Thread.sleep(3000);
            plt.clear();

            ArrayList<Double> x = new ArrayList<>();
            for(double i = -1.0; i <= 2.0; i += 0.01)
                x.add(i);

            double[][] array = new double[x.size()][2];
            for(int i = 0; i < x.size(); i++) {
                array[i][0] = x.get(i);
                array[i][1] = demo.F(x.get(i), p);
            }

            Line line = new Line(array, Line.Style.SOLID, ' ', Color.BLUE);
            var lpt = new LinePlot(line).canvas();
            var ubd = lpt.getUpperBounds();
            ubd[1] = 1.01;
            lpt.extendUpperBound(ubd);                     // set y limit
            lpt.setTitle("Bernoulli");
            lpt.setAxisLabel(0, "x");
            lpt.setAxisLabel(1, "c.d.f.");
            lpt.window();
            Thread.sleep(3000);
            lpt.clear();

            // ---------------------------------
            // Discrete Uniform
            // ---------------------------------
            double n = 5.0;
            double[][] ddt = new double[(int)n][2];
            for(int i = 1; i <= n; i++) {
                ddt[i-1][0] = i * 1.0;
                ddt[i-1][1] = (1/n);
            }

            Bar dbar = new Bar(ddt, 0.05, Color.BLUE);
            var dplt = new BarPlot(dbar).canvas();
            var lbd = dplt.getLowerBounds();
            lbd[0] = 0.0;
            lbd[1] = -0.01;
            ubd = dplt.getUpperBounds();
            ubd[0] = 6.0;
            ubd[1] = 0.5;
            dplt.extendLowerBound(lbd);
            dplt.extendUpperBound(ubd);
            dplt.setTitle("Discrete Uniform");
            dplt.setAxisLabels("x", "p.m.f.");
            StatisticHelper.printVectorElements(dplt.getLowerBounds());
            StatisticHelper.printVectorElements(dplt.getUpperBounds());
            dplt.window();
            Thread.sleep(3000);
            dplt.clear();

            // Now, let us plot the cumulative distribution function
            x.clear();
            for(double i = -1.0; i <= 6.0; i += 0.01)
                x.add(i);

            double[][] darray = new double[x.size()][2];
            for(int i = 0; i < x.size(); i++) {
                darray[i][0] = x.get(i);
                darray[i][1] = demo.DF(x.get(i), n);
            }

            Line dline = new Line(darray, Line.Style.SOLID, ' ', Color.BLUE);
            var dlpt = new LinePlot(dline).canvas();
            ubd = dlpt.getUpperBounds();
            ubd[1] = 1.01;
            dlpt.extendUpperBound(ubd);
            dlpt.setTitle("Discrete Uniform");
            dlpt.setAxisLabel(0, "x");
            dlpt.setAxisLabel(1, "c.d.f.");
            dlpt.window();
            Thread.sleep(3000);
            dlpt.clear();

            // ----------------------------------
            // Continuous Uniform
            // ----------------------------------
            double a = 1.0, b = 3.0;
            x.clear();
            ArrayList<Double> y = new ArrayList<>();
            for(double i = 0.0; i <= 4.0; i += 0.01) {
                x.add(i);
                if( i > a && i < b)
                    y.add(1.0);
                else
                    y.add(0.0);
            }
            double[][] cdt = new double[x.size()][2];
            for(int i = 0; i < x.size(); i++) {
                cdt[i][0] = x.get(i);
                cdt[i][1] = y.get(i);
            }

            Line cline = new Line(cdt, Line.Style.SOLID, ' ', Color.BLUE);
            var clpt = new LinePlot(cline).canvas();
            StatisticHelper.printVectorElements(clpt.getLowerBounds());
            lbd = clpt.getLowerBounds();
            lbd[1] = -0.01;
            ubd = clpt.getUpperBounds();
            ubd[1] = 1.01;
            clpt.extendBound(lbd, ubd);
            clpt.setTitle("Continuous Uniform");
            clpt.setAxisLabels("x", "p.m.f.");
            clpt.window();
            Thread.sleep(3000);
            clpt.clear();

            // Now, let us plot the cumulative distribution function
            for(int i = 0; i < x.size(); i++) {
                cdt[i][1] = demo.CF(x.get(i), a, b);
            }
            cline = new Line(cdt, Line.Style.SOLID, ' ', Color.BLUE);
            clpt = new LinePlot(cline).canvas();
            StatisticHelper.printVectorElements(clpt.getLowerBounds());
            lbd = clpt.getLowerBounds();
            lbd[1] = -0.01;
            ubd = clpt.getUpperBounds();
            ubd[1] = 1.01;
            clpt.extendBound(lbd, ubd);
            clpt.setTitle("Continuous Uniform");
            clpt.setAxisLabels("x", "c.d.f.");
            clpt.window();
            Thread.sleep(3000);
            clpt.clear();

            // -----------------------------
            // Binomial
            // -----------------------------
            int N = 10;
            p = 0.2;
            x.clear();
            y.clear();
            for( int i = 0; i < (N + 1); i++ ) {
                x.add(i*1.0);
                double t = Math.pow(p, i) * Math.pow((1-p), (N-i)) * demo.binom(N, i).longValue();
                y.add(t);
            }

            double[][] bdt = new double[x.size()][2];
            for(int i = 0; i < x.size(); i++) {
                bdt[i][0] = x.get(i);
                bdt[i][1] = y.get(i);
            }

            Bar bbar = new Bar(bdt, 0.1, Color.BLUE);
            var bplt = new BarPlot(bbar).canvas();
            bplt.setTitle("Binomial");
            bplt.setAxisLabels("x", "p.m.f.");
            bplt.window();
            Thread.sleep(3000);
            bplt.clear();

            ArrayList<Double> xx = new ArrayList<>();
            for(double i = -1.0; i <= 11.0; i += 0.01)
                xx.add(i);
            ArrayList<Double> cmf = StatisticHelper.cumulativeSum(y);

            double[][] bfdt = new double[xx.size()][2];
            for( int i = 0; i < xx.size(); i++ ) {
                bfdt[i][0] = xx.get(i);
                bfdt[i][1] = demo.BF(xx.get(i), N*1.0, cmf);
            }

            var bline = new Line(bfdt, Line.Style.SOLID, ' ', Color.BLUE);
            var blpt = new LinePlot(bline).canvas();
            StatisticHelper.printVectorElements(blpt.getLowerBounds());
            lbd = blpt.getLowerBounds();
            lbd[1] = -0.01;
            ubd = blpt.getUpperBounds();
            ubd[1] = 1.01;
            blpt.extendBound(lbd, ubd);
            blpt.setTitle("Binomial");
            blpt.setAxisLabels("x", "c.d.f.");
            blpt.window();
            Thread.sleep(3000);
            blpt.clear();

            // ---------------------------
            // Poisson
            // ---------------------------
            double lam = 5.0;

            System.out.println( gamma(2 + 1) );
            y.clear();
            double[][] pdt = new double[20][2];
            for( int i = 0; i < 20; i++ ) {
                pdt[i][0] = (i*1.0);
                var t = (Math.exp(-1.0*lam)*Math.pow(lam, i)/gamma(i + 1));
                pdt[i][1] = t;
                y.add(t);
            }

            Bar pbar = new Bar(pdt, 0.1, Color.BLUE);
            var pplt = new BarPlot(pbar).canvas();
            pplt.setTitle("Poisson");
            pplt.setAxisLabels("x", "p.m.f.");
            pplt.window();
            Thread.sleep(3000);
            pplt.clear();

            cmf = StatisticHelper.cumulativeSum((ArrayList<Double>) y.clone());
            x.clear();
            y.clear();
            for(double i = -1.0; i <= 21.0; i += 0.01) {
                x.add(i);
                y.add(demo.BF(i, N*1.0, cmf));
            }

            double[][] ppdt = new double[x.size()][2];
            for( int i = 0; i < x.size(); i++ ) {
                ppdt[i][0] = x.get(i);
                ppdt[i][1] = y.get(i);
            }

            var pline = new Line(ppdt, Line.Style.SOLID, ' ', Color.BLUE);
            var pplpt = new LinePlot(pline).canvas();
            lbd = pplpt.getLowerBounds();
            lbd[1] = -0.01;
            ubd = pplpt.getUpperBounds();
            ubd[1] = 1.01;
            pplpt.extendBound(lbd, ubd);
            pplpt.setTitle("Poisson");
            pplpt.setAxisLabels("x", "c.d.f.");
            pplpt.window();
            Thread.sleep(3000);
            pplpt.clear();

            // ---------------------------
            // Gaussian
            // ---------------------------
            // Let us first plot the probability density function
            double mu = 0.0, sigma = 1.0;
            x.clear();
            y.clear();
            for(double i = -3.0; i <= 3.0; i += 0.01) {
                x.add(i);
                p = 1/Math.sqrt(2*Math.PI*Math.pow(sigma, 2))
				* Math.exp(-1.0*Math.pow((i-mu),2.0)/(2*Math.pow(sigma, 2)));
                y.add(p);
            }

            double[][] gdt = new double[x.size()][2];
            for( int i = 0; i < x.size(); i++ ) {
                gdt[i][0] = x.get(i);
                gdt[i][1] = y.get(i);
            }

            var gline = new Line(gdt, Line.Style.SOLID, ' ', Color.BLUE);
            var glpt = new LinePlot(gline).canvas();
            lbd = glpt.getLowerBounds();
            lbd[0] = -3.0;
            lbd[1] = -0.01;
            ubd = glpt.getUpperBounds();
            ubd[0] = 3.0;
            ubd[1] = 0.5;
            glpt.extendBound(lbd, ubd);
            glpt.setTitle("Gaussian");
            glpt.setAxisLabels("x", "p.d.f.");
            glpt.window();
            Thread.sleep(3000);
            glpt.clear();

            // ----------------------------------
            // Phi - density function for a standard normal distribution
            // ----------------------------------
            double[][] phidt = new double[x.size()][2];
            for( int i = 0; i < x.size(); i++ ) {
                phidt[i][0] = x.get(i);
                phidt[i][1] = demo.phi(x.get(i), mu, sigma);
            }
            var phil = new Line(phidt, Line.Style.SOLID, ' ', Color.BLUE);
            var philpt = new LinePlot(phil).canvas();
            lbd = philpt.getLowerBounds();
            lbd[0] = -3.0;
            lbd[1] = -0.01;
            ubd = philpt.getUpperBounds();
            ubd[0] = 3.0;
            ubd[1] = 1.01;
            philpt.extendBound(lbd, ubd);
            philpt.setTitle("Cumulative distribution function");
            philpt.setAxisLabels("x", "c.d.f.");
            philpt.window();
            Thread.sleep(3000);
            philpt.clear();

        } catch(Exception e) {
            e.printStackTrace();
        }
    }
}
