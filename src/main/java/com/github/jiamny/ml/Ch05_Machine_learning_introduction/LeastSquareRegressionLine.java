package com.github.jiamny.ml.Ch05_Machine_learning_introduction;

import java.util.Arrays;

public class LeastSquareRegressionLine {

    // Function to calculate b
    private static double calculateB(int[] x, int[] y) {
        int n = x.length;

        // sum of array x
        int sx = Arrays.stream(x).sum();

        // sum of array y
        int sy = Arrays.stream(y).sum();

        // for sum of product of x and y
        int sxsy = 0;

        // sum of square of x
        int sx2 = 0;
        for (int i = 0; i < n; i++) {
            sxsy += x[i] * y[i];
            sx2 += x[i] * x[i];
        }
        double b = (double)(n * sxsy - sx * sy)
                / (n * sx2 - sx * sx);

        return b;
    }

    // Function to find the
    // least regression line
    public static void leastRegLine(int X[], int Y[]) {

        // Finding b
        double b = calculateB(X, Y);

        int n = X.length;
        int meanX = Arrays.stream(X).sum() / n;
        int meanY = Arrays.stream(Y).sum() / n;

        // calculating a
        double a = meanY - b * meanX;

        // Printing regression line
        System.out.println("Regression line:");
        System.out.print("Y = ");
        System.out.printf("%.3f", a);
        System.out.print(" + ");
        System.out.printf("%.3f", b);
        System.out.print("*X");
    }

    // Driver code
    public static void main(String[] args) {
        // statistical data
        int X[] = { 95, 85, 80, 70, 60 };
        int Y[] = { 90, 80, 70, 65, 60 };

        leastRegLine(X, Y);
    }
}
