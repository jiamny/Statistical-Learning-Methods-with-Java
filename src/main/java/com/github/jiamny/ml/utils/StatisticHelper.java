package com.github.jiamny.ml.utils;

import java.util.*;
import java.util.function.Function;
import smile.math.MathEx;

import static com.github.jiamny.ml.utils.DataFrameHelper.range;

public class StatisticHelper {
    public static <T extends Comparable<T>> T mode(T[] a, T initV) {
        T maxValue = initV;
        int maxCount = 0, i, j;
        int n = a.length;

        for (i = 0; i < n; ++i) {
            int count = 0;
            for (j = 0; j < n; ++j) {
                if (a[j].compareTo(a[i]) == 0)
                    ++count;
            }

            if (count > maxCount) {
                maxCount = count;
                maxValue = a[i];
            }
        }
        return maxValue;
    }

    public static void printVectorElements(double[] ax) {
        Arrays.stream(ax).forEach(num -> System.out.print(num + " "));
        System.out.println();
    }

    public static void printVectorElements(double[] ax, int decimal) {
        Arrays.stream(ax).forEach(num -> System.out.printf(String.valueOf("%." + decimal + "f "), num));
        System.out.println();
    }

    public static void printVectorElements(int[] ax) {
        Arrays.stream(ax).forEach(num -> System.out.print(num + " "));
        System.out.println();
    }

    public static void printVectorObjects(Object[] ax) {
        Arrays.stream(ax).forEach(num -> System.out.print(num + " "));
        System.out.println();
    }

    public static ArrayList<Double> cumulativeSum(ArrayList<Double> numbers) {
        // variable
        double sum = 0.0;

        // traverse through the array
        for (int i = 0; i < numbers.size(); i++) {
            sum += numbers.get(i); // find sum
            numbers.set(i, sum);   // replace
        }
        // return
        return numbers;
    }

    public static int [] cumSum(int [] numbers) {
        // variable
        int sum = 0;

        // traverse through the array
        for (int i = 0; i < numbers.length; i++) {
            sum += numbers[i];  // find sum
            numbers[i] = sum;   // replace
        }
        // return
        return numbers;
    }

    public static <K, V> Map.Entry<K, V> maxOrmin(Map<K, V> map, Comparator<V> comp, boolean max) {
        Iterator<Map.Entry<K, V>> entries = map.entrySet().iterator();

        if (!entries.hasNext()) {
            return null;
        }
        Map.Entry<K, V> mv;
        for (mv = entries.next(); entries.hasNext(); ) {
            Map.Entry<K, V> value = entries.next();
            if (max) {
                if (comp.compare(value.getValue(), mv.getValue()) > 0) {
                    mv = value;
                }
            } else {
                if (comp.compare(value.getValue(), mv.getValue()) < 0) {
                    mv = value;
                }
            }

        }
        return mv;
    }

    public static double[] linespace(double min, double max, int points) {
        double[] d = new double[points];
        for (int i = 0; i < points; i++) {
            d[i] = min + i * (max - min) / (points - 1);
        }
        return d;
    }

    public static double[][] impute(Function<double[][], double[][]> imputer, double[][] data) throws Exception {
        MathEx.setSeed(19650218); // to get repeatable results.
/*
        int n = 0;
        double[][] missing = new double[data.length][data[0].length];
        for (int i = 0; i < missing.length; i++) {
            for (int j = 0; j < missing[i].length; j++) {
                if (MathEx.random() < rate) {
                    n++;
                    missing[i][j] = Double.NaN;
                } else {
                    missing[i][j] = data[i][j];
                }
            }
        }
*/
        double[][] imputed = imputer.apply(data);

        double error = 0.0;
        for (int i = 0; i < imputed.length; i++) {
            for (int j = 0; j < imputed[i].length; j++) {
                error += Math.abs(data[i][j] - imputed[i][j]) / data[i][j];
            }
        }
        System.out.println("error: " + error);
        return imputed;
    }

    public static int maxIndex(double[] data) {
        int idx = -1;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < data.length; i++) {
            if (max < data[i]) {
                idx = i;
                max = data[i];
            }
        }
        return idx;
    }

    public static int minIndex(double[] data) {
        int idx = -1;
        double min = Double.POSITIVE_INFINITY;
        for (int i = 0; i < data.length; i++) {
            if (min > data[i]) {
                idx = i;
                min = data[i];
            }
        }
        return idx;
    }

    /**
     * x > 0 => 1; x == 0 => 0; x < 0 => -1
     *
     * @param x
     * @return
     */
    public static int sign(double x) {
        if (x > 0)
            return 1;
        else if (x < 0)
            return -1;
        else
            return 0;
    }

    public static double[] ShuffleArray(double[] array) {
        Random rand = new Random();  // Random value generator
        for (int i = 0; i < array.length; i++) {
            int randomIndex = rand.nextInt(array.length);
            double temp = array[i];
            array[i] = array[randomIndex];
            array[randomIndex] = temp;
        }
        return array;
    }

    public static double sum(double[] ax) {
        double sum = 0.0;
        for(int i = 0; i < ax.length; i++)
            sum += ax[i];
        return sum;
    }

    public static long sum(int[] ax) {
        long sum = 0;
        for(int i = 0; i < ax.length; i++)
            sum += ax[i];
        return sum;
    }

    public static void helper(List<int[]> combinations, int data[], int start, int end, int index) {
        if (index == data.length) {
            int[] combination = data.clone();
            combinations.add(combination);
        } else if (start <= end) {
            data[index] = start;
            helper(combinations, data, start + 1, end, index + 1);
            helper(combinations, data, start + 1, end, index);
        }
    }

    public static List<int[]> combGenerate(int n, int r) {
        List<int[]> combinations = new ArrayList<>();
        helper(combinations, new int[r], 0, n-1, 0);
        return combinations;
    }

    public static HashMap<Integer, Double> sortByValue(HashMap<Integer, Double> hm)
    {
        // Create a list from elements of HashMap
        List<Map.Entry<Integer, Double> > list =
                new LinkedList<Map.Entry<Integer, Double> >(hm.entrySet());

        // Sort the list
        Collections.sort(list, new Comparator<Map.Entry<Integer, Double> >() {
            public int compare(Map.Entry<Integer, Double> o1,
                               Map.Entry<Integer, Double> o2)
            {
                return (o1.getValue()).compareTo(o2.getValue());
            }
        });

        // put data from sorted list to hashmap
        HashMap<Integer, Double> temp = new LinkedHashMap<Integer, Double>();
        for (Map.Entry<Integer, Double> aa : list) {
            temp.put(aa.getKey(), aa.getValue());
        }
        return temp;
    }

    public static HashMap<Integer, Double> sortByValueDesc(HashMap<Integer, Double> hm)
    {
        // Create a list from elements of HashMap
        List<Map.Entry<Integer, Double> > list =
                new LinkedList<Map.Entry<Integer, Double> >(hm.entrySet());

        // Sort the list
        Collections.sort(list, new Comparator<Map.Entry<Integer, Double> >() {
            public int compare(Map.Entry<Integer, Double> o1,
                               Map.Entry<Integer, Double> o2)
            {
                return (o2.getValue()).compareTo(o1.getValue());
            }
        });

        // put data from sorted list to hashmap
        HashMap<Integer, Double> temp = new LinkedHashMap<Integer, Double>();
        for (Map.Entry<Integer, Double> aa : list) {
            temp.put(aa.getKey(), aa.getValue());
        }
        return temp;
    }

    public static int [] argSort(double [] d) {
        HashMap<Integer, Double> hm = new HashMap<>();
        for(int i : range(d.length)) {
            hm.put(i, d[i]);
        }
        hm = sortByValue(hm);
        List<Map.Entry<Integer, Double> > list =
                new LinkedList<Map.Entry<Integer, Double> >(hm.entrySet());
        int [] idx = new int[d.length];
        int i = 0;
        for (Map.Entry<Integer, Double> aa : list) {
            idx[i] = aa.getKey();
            i++;
        }
        return idx;
    }
}
