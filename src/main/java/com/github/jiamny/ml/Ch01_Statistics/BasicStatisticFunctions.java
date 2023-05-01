package com.github.jiamny.ml.Ch01_Statistics;

import com.github.jiamny.ml.utils.StatisticHelper;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import org.apache.commons.math3.stat.descriptive.moment.Kurtosis;
import org.apache.commons.math3.stat.descriptive.moment.Skewness;

import static smile.math.MathEx.*;

public class BasicStatisticFunctions {

    public static void main(String [] args) {
        Double[] x = {40.0, 45.0, 23.0, 39.0, 39.0};
        // Mean
        System.out.println("mean: " + mean(ArrayUtils.toPrimitive(x)));

        // Mode
        System.out.println("mode: " + StatisticHelper.mode(x, 0.0));

        // Median
        System.out.println("median: " + median(ArrayUtils.toPrimitive(x)));

        // Variance
        System.out.println("Variance: " + var(ArrayUtils.toPrimitive(x)));

        // Standard deviation
        System.out.println("Standard deviation: " + sd(ArrayUtils.toPrimitive(x)));

        // Measuring dispersion
        System.out.println("min: " + min(ArrayUtils.toPrimitive(x)) + ", max: " + max(ArrayUtils.toPrimitive(x)));

        DescriptiveStatistics da = new DescriptiveStatistics(ArrayUtils.toPrimitive(x));
        double iqr = da.getPercentile(75) - da.getPercentile(25);
        System.out.println("First Quartile: " + da.getPercentile(25) + ", Third Quartile: " + da.getPercentile(75));
        System.out.println("iqr: " + iqr);

        // Skewness and kurtosis
        Skewness skewness = new Skewness();
        System.out.println("skewness: " + skewness.evaluate(ArrayUtils.toPrimitive(x)));

        /*
        A normal distribution having zero kurtosis is known as a mesokurtic distribution.
        A platykurtic distribution has a negative kurtosis value and is thin-tailed compared to a normal distribution.
        A leptokurtic distribution has a kurtosis value greater than 3 and is fat-tailed compared to a normal distribution.
         */
        Kurtosis kurtosis = new Kurtosis();
        System.out.println("kurtosis: " + kurtosis.evaluate(ArrayUtils.toPrimitive(x)));

        // covariance and correlation coefficients
        /*
        pearson: Standard correlation coefficient
        kendall: Kendall's tau correlation coefficient - a type of rank correlation.
                 It measures the similarity or dissimilarity between two variables.
        spearman: Spearman's rank correlation coefficient - Pearson's correlation coefficient on the ranks of the observations.
                 It assesses the strength of the association between two ranked variables.
         */
        double [] y = {38.0, 41.0, 42.0, 48.0, 32.0};
        // covariance
        System.out.println("cov(x, y): " + cov(ArrayUtils.toPrimitive(x), y));

        // pearson
        System.out.println("cor(x, y): " + cor(ArrayUtils.toPrimitive(x), y));

        // Spearman
        System.out.println("Spearman(x, y): " + spearman(ArrayUtils.toPrimitive(x), y));

        // Kendall's
        System.out.println("Kendall(x, y): " + kendall(ArrayUtils.toPrimitive(x), y));
    }
}
