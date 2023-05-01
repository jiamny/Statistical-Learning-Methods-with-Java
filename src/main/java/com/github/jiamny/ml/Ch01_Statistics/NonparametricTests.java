package com.github.jiamny.ml.Ch01_Statistics;

import smile.stat.hypothesis.ChiSqTest;

public class NonparametricTests {

    public static void main(String [] args) {
        /*
        A Chi-Square test is determined by a significant difference or relationship
        between two categorical variables from a single population.

        Null Hypothesis H0: The two categorical variables are independent—that is,
            employee performance is independent of the highest qualification level.
        Alternative Hypothesis Ha: The two categorical variables are not independent—that is,
            employee performance is not independent of the highest qualification level.
         */
        // Average performing employees
        //int [] average = {20, 16, 13, 7};
        // Outstanding performing employees
        //int []  outstanding = {31, 40, 60, 13};
        int [][] tb = {{20, 16, 13, 7}, {31, 40, 60, 13}};

        // Given a two-dimensional contingency table
        ChiSqTest cqt = ChiSqTest.test(tb);

        System.out.println("ChiSq value: " + cqt.chisq);
        System.out.println("pvalue: " + cqt.pvalue);

        // 0.05 or 5% is significance level or alpha.
        if( cqt.pvalue < 0.05 )
            System.out.println("Hypothesis Rejected, categorical variables are not independent");
        else
            System.out.println("Hypothesis Accepted, categorical variables are independent with a 95% confidence interval");

        // One-Sample Test
        int[] bins = {20, 22, 13, 22, 10, 13};
        double[] prob = {1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6};
        cqt = ChiSqTest.test(bins, prob);

        System.out.println("\nChiSq value: " + cqt.chisq);
        System.out.println("pvalue: " + cqt.pvalue);

        if( cqt.pvalue < 0.05 )
            System.out.println("Hypothesis Rejected, categorical variables are not independent");
        else
            System.out.println("Hypothesis Accepted, categorical variables are independent with a 95% confidence interval");

        // Two-Sample Test
        int[] bins1 = {8, 13, 16, 10, 3};
        int[] bins2 = {4,  9, 14, 16, 7};
        System.out.println("\n" + ChiSqTest.test(bins1, bins2));

        // Independence Test
        int[][] x = { {12, 7}, {5, 7} };
        System.out.println("\n" + ChiSqTest.test(x));
    }
}
