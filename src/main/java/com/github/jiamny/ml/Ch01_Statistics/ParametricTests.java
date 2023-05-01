package com.github.jiamny.ml.Ch01_Statistics;

import smile.stat.distribution.GaussianDistribution;
import smile.stat.hypothesis.FTest;
import smile.stat.hypothesis.KSTest;
import smile.stat.hypothesis.TTest;

import static smile.math.MathEx.mean;

public class ParametricTests {

    public static void main(String [] args) {
        double [] data = {63, 75, 84, 58, 52, 96, 63, 55, 76, 83};
        // mean
        System.out.println("mean: " + mean(data));

        // a one-sample t-test
        TTest tst = TTest.test(data, 68.0);
        System.out.println("t value: " + tst.t);
        System.out.println("pvalue: " + tst.pvalue);

        // 0.05 or 5% is significance level or alpha.
        if( tst.pvalue < 0.05 )
            System.out.println("Hypothesis Rejected, mu is not 68.0");
        else
            System.out.println("Hypothesis Accepted, mu should be 68.0 with a 95% confidence interval");

        /*
        A two-sample t-test

        Null Hypothesis H0: Sample means are equal—μ 1 = μ 2
        Alternative Hypothesis Ha: Sample means are not equal—μ 1 > μ 2 or μ 2 > μ 1
         */
        double [] a1 = {63, 75, 84, 58, 52, 96, 63, 55, 76, 83};
        double [] a2 = {53, 43, 31, 113, 33, 57, 27, 23, 24, 43};

        tst = TTest.test(a1, a2);
        System.out.println("\nt value: " + tst.t);
        System.out.println("pvalue: " + tst.pvalue);

        // 0.05 or 5% is significance level or alpha.
        if( tst.pvalue < 0.05 )
            System.out.println("Hypothesis Rejected, a1 and a2 has different mu");
        else
            System.out.println("Hypothesis Accepted, a1 and a2 has the same mu with a 95% confidence interval");

        /*
        A paired sample t-test is a dependent sample t-test, which is used to decide
        whether the mean difference between two observations of the same group is zero.

        Null Hypothesis H0: Mean difference between the two dependent samples is 0.
        Alternative Hypothesis Ha: Mean difference between the two dependent samples is not 0.
         */
        double [] p1 = {63, 75, 84, 58, 52, 96, 63, 65, 76, 83};
        double [] p2 = {53, 43, 67, 59, 48, 57, 65, 58, 64, 72};

        tst = TTest.testPaired(p1, p2);
        System.out.println("\nt value: " + tst.t);
        System.out.println("pvalue: " + tst.pvalue);

        // 0.05 or 5% is significance level or alpha.
        if( tst.pvalue < 0.05 )
            System.out.println("Hypothesis Rejected, mean difference between the two dependent samples is 0");
        else
            System.out.println("Hypothesis Accepted, mean difference between the two dependent samples is not 0 with a 95% confidence interval");

        // one-way ANOVA test
        /*
        Null Hypothesis H0: There is no difference between the mean performance of multiple locations.
        Alternative Hypothesis Ha: There is a difference between the mean performance score of multiple locations.
        */
        double [] mumbai = {0.14730927, 0.59168541, 0.85677052, 0.27315387,
                0.78591207,0.52426114, 0.05007655, 0.64405363, 0.9825853 ,
                0.62667439};
        double [] chicago = {0.99140754, 0.76960782, 0.51370154, 0.85041028,
                0.19485391,0.25269917, 0.19925735, 0.80048387, 0.98381235,
                0.5864963 };

        FTest ftst = FTest.test(mumbai, chicago);
        System.out.println("\nF value: " + ftst.f);
        System.out.println("pvalue: " + ftst.pvalue);

        // 0.05 or 5% is significance level or alpha.
        if( ftst.pvalue < 0.05 )
            System.out.println("Hypothesis Rejected, no difference between the mean performance");
        else
            System.out.println("Hypothesis Accepted, there is a difference between the mean performance score with a 95% confidence interval");

        // Kolmogorov–Smirnov Test for the null hypothesis that the data set x is drawn from the given distribution.
        double [] x = {
                0.53236606, -1.36750258, -1.47239199, -0.12517888, -1.24040594, 1.90357309,
                -0.54429527, 2.22084140, -1.17209146, -0.68824211, -1.75068914, 0.48505896,
                2.75342248, -0.90675303, -1.05971929, 0.49922388, -1.23214498, 0.79284888,
                0.85309580, 0.17903487, 0.39894754, -0.52744720, 0.08516943, -1.93817962,
                0.25042913, -0.56311389, -1.08608388, 0.11912253, 2.87961007, -0.72674865,
                1.11510699, 0.39970074, 0.50060532, -0.82531807, 0.14715616, -0.96133601,
                -0.95699473, -0.71471097, -0.50443258, 0.31690224, 0.04325009, 0.85316056,
                0.83602606, 1.46678847, 0.46891827, 0.69968175, 0.97864326, 0.66985742,
                -0.20922486, -0.15265994};

        System.out.println( "\n" + KSTest.test(x, new GaussianDistribution(0, 1)));

        // Two-Sample Test
        double []  y = {
                0.95791391, 0.16203847, 0.56622013, 0.39252941, 0.99126354, 0.65639108,
                0.07903248, 0.84124582, 0.76718719, 0.80756577, 0.12263981, 0.84733360,
                0.85190907, 0.77896244, 0.84915723, 0.78225903, 0.95788055, 0.01849366,
                0.21000365, 0.97951772, 0.60078520, 0.80534223, 0.77144013, 0.28495121,
                0.41300867, 0.51547517, 0.78775718, 0.07564151, 0.82871088, 0.83988694};

        System.out.println( "\n" + KSTest.test(x, y));

    }
}
