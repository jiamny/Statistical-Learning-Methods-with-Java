package com.github.jiamny.ml.Ch04_Data_preprocessing;


import org.apache.commons.csv.CSVFormat;
import org.apache.commons.lang3.ArrayUtils;
import smile.base.cart.SplitRule;
import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.formula.Formula;
import smile.data.type.DataType;
import smile.data.vector.BaseVector;
import smile.data.vector.DoubleVector;
import smile.data.vector.IntVector;
import smile.data.vector.StringVector;
import smile.feature.imputation.*;
import smile.io.Read;
import java.util.function.Function;
import smile.math.MathEx;
import smile.math.distance.Distance;
import smile.math.matrix.Matrix;

import java.util.Arrays;
import java.util.List;
import java.util.stream.BaseStream;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import static com.github.jiamny.ml.utils.DataFrameHelper.*;
import static java.lang.Double.NaN;
import static smile.math.MathEx.mean;
import static smile.math.MathEx.sd;

import com.github.jiamny.ml.utils.StatisticHelper;

public class DataPreprocessing {

    private static   long[] seeds = {
            342317953, 521642753, 72070657, 577451521, 266953217, 179976193,
            374603777, 527788033, 303395329, 185759582, 261518209, 461300737,
            483646580, 532528741, 159827201, 284796929, 655932697, 26390017,
            454330473, 867526205, 824623361, 719082324, 334008833, 699933293,
            823964929, 155216641, 150210071, 249486337, 713508520, 558398977,
            886227770, 74062428, 670528514, 701250241, 363339915, 319216345,
            757017601, 459643789, 170213767, 434634241, 414707201, 153100613,
            753882113, 546490145, 412517763, 888761089, 628632833, 565587585,
            175885057, 594903553, 78450978, 212995578, 710952449, 835852289,
            415422977, 832538705, 624345857, 839826433, 260963602, 386066438,
            530942946, 261866663, 269735895, 798436064, 379576194, 251582977,
            349161809, 179653121, 218870401, 415292417, 86861523, 570214657,
            701581299, 805955890, 358025785, 231452966, 584239408, 297276298,
            371814913, 159451160, 284126095, 896291329, 496278529, 556314113,
            31607297, 726761729, 217004033, 390410146, 70173193, 661580775,
            633589889, 389049037, 112099159, 54041089, 80388281, 492196097,
            912179201, 699398161, 482080769, 363844609, 286008078, 398098433,
            339855361, 189583553, 697670495, 709568513, 98494337, 99107427,
            433350529, 266601473, 888120086, 243906049, 414781441, 154685953,
            601194298, 292273153, 212413697, 568007473, 666386113, 712261633,
            802026964, 783034790, 188095005, 742646355, 550352897, 209421313,
            175672961, 242531185, 157584001, 201363231, 760741889, 852924929,
            60158977, 774572033, 311159809, 407214966, 804474160, 304456514,
            54251009, 504009638, 902115329, 870383757, 487243777, 635554282,
            564918017, 636074753, 870308031, 817515521, 494471884, 562424321,
            81710593, 476321537, 595107841, 418699893, 315560449, 773617153,
            163266399, 274201241, 290857537, 879955457, 801949697, 669025793,
            753107969, 424060977, 661877468, 433391617, 222716929, 334154852,
            878528257, 253742849, 480885528, 99773953, 913761493, 700407809,
            483418083, 487870398, 58433153, 608046337, 475342337, 506376199,
            378726401, 306604033, 724646374, 895195218, 523634541, 766543466,
            190068097, 718704641, 254519245, 393943681, 796689751, 379497473,
            50014340, 489234689, 129556481, 178766593, 142540536, 213594113,
            870440184, 277912577};

    public static void main( String [] args ) {

        double[][] A = {
                {1.0,2.0,3.0,4.0},
                {5.0,6.0,NaN,8.0},
                {10.0,11.0,12.0, NaN}
        };

        DataFrame dtA = DataFrame.of(A, new String[]{"A", "B", "C", "D"});
        boolean hasMissing = hasDataMissing(dtA.toMatrix());
        System.out.println(hasMissing);

        if( hasMissing ) {
            try {
                // ---------------------------------------------------
                // Dealing with missing data
                // ---------------------------------------------------
                System.out.println(dtA);
                System.out.println(">>> Eliminating examples or features with missing values");
                // Eliminating training examples or features with missing values
                System.out.println("------------------- Eliminating features with missing values");
                var datC = dtA.drop(2,3);
                System.out.println(datC);

                System.out.println("------------------- Eliminating features with missing values");
                var dtR = dtA.omitNullRows();
                System.out.println(dtR);

                System.out.println(">>> Imputing missing values");
                // Average Value Imputation
                System.out.println("------------------- Average Value Imputation");
                System.out.println(dtA);
                var B = dtA.toArray();
                B = SimpleImputer.impute(B);
                var dtB = DataFrame.of(B, new String[]{"A", "B", "C", "D"});
                System.out.println(dtB);

                // K-Nearest Neighbor Imputation
                System.out.println("------------------- K-Nearest Neighbor Imputation");
                var C = dtA.toArray();

                KNNImputer knnIpt = new KNNImputer(dtA, 2);
                System.out.println(dtA);
                Function<double[][], double[][]> imputer = x -> knnIpt.apply(DataFrame.of(x)).toArray();
                C = StatisticHelper.impute(imputer, C);
                var dtC = DataFrame.of(C, new String[]{"A", "B", "C", "D"});
                System.out.println(dtC);
/*
                // K-Means Imputation
                System.out.println("------------------- K-Means Imputation");
                Distance<Tuple> distance = (x, y) -> {
                    double[] xd = x.toArray();
                    double[] yd = y.toArray();
                    return MathEx.squaredDistanceWithMissingValues(xd, yd);
                };

                var D = dtA.toArray();
                KMedoidsImputer kmedoidsImputer = KMedoidsImputer.fit(dtA, distance,2);
                imputer = x -> kmedoidsImputer.apply(DataFrame.of(x)).toArray();
                System.out.println(dtA);
                D = StatisticHelper.impute(imputer, D);
                var dtD = DataFrame.of(D, new String[]{"A", "B", "C", "D"});
                System.out.println(dtD);

                // SVD Imputation
                System.out.println("------------------- SVD Imputation");
                var F = dtA.toArray();
                int k = F[0].length / 5;
                imputer = x -> SVDImputer.impute(x, k, 3);
                System.out.println(dtA);

                StatisticHelper.impute(imputer, F);
                var dtF = DataFrame.of(F, new String[]{"A", "B", "C", "D"});
                System.out.println(dtF);
*/
                // ------------------------------------------
                // Handling categorical data
                // ------------------------------------------
                BaseVector[] bv = new BaseVector[]{
                        StringVector.of("color", new String[]{"green", "red", "blue"}),
                        StringVector.of("size", new String[]{"M", "L", "XL"}),
                        DoubleVector.of("price", new double[]{11.1, 13.5, 15.3}),
                        StringVector.of("class", new String[]{"class2", "class1", "class2"})
                };
                var df = DataFrame.of(bv);
                System.out.println(df);

                // Mapping ordinal features
                System.out.println("------------------- Mapping ordinal features");
                var newdf = df.drop("size");
                var mdf = newdf.merge(IntVector.of("size", new int[]{1, 2, 3}));
                System.out.println(mdf);

                //  Encoding class labels
                System.out.println("------------------- Encoding class labels");
                StringVector sz = (StringVector) df.column("size");
                System.out.println(sz);

                int [] v1 = new int[]{0, 0, 0};
                int [] v2 = new int[]{0, 0, 0};
                for( int i = 0; i < mdf.nrow(); i++ ) {
                    if( ! sz.get(i).equalsIgnoreCase("M") )  v1[i] = 1;
                    if( sz.get(i).equalsIgnoreCase("XL") ) v2[i] = 1;
                }

                BaseVector[] ecv = new BaseVector[]{
                        IntVector.of("X > M", v1),
                        IntVector.of("X > L", v2),
                };
                newdf = df.drop("size");
                mdf = newdf.merge(ecv);
                System.out.println(mdf);

                // Partitioning a dataset into a seperate training and test set
                var format = CSVFormat.newFormat(',');
                var wineDt = Read.csv("./data/wine.data", format);

                System.out.println("------------------- Partitioning a dataset");
                String [] names = new String[] {
                        "class", "Alcohol", "MalicAcid", "Ash", "AlcalinityOfAsh", "Magnesium", "TotalPhenols",
                        "Flavanoids", "NonflavanoidPhenols", "Proanthocyanins", "ColorIntensity", "Hue",
                        "OD280_OD315_of_diluted_wines", "Proline"};
                wineDt = DataFrame.of(wineDt.toArray(), names);
                System.out.println(wineDt);
                double train_ratio = 0.7;
                int train_sz = (int)(train_ratio*wineDt.nrow());
                System.out.println(train_sz);
                Matrix dt = wineDt.toMatrix();
                System.out.println(dt);

                var train_dt = dt.rows(range(train_sz));
                System.out.println("matrix size: " + train_dt.nrow() + " " + train_dt.ncol());

                // -----------------------------------------------
                // Bringing features onto the same scale
                // -----------------------------------------------
                var ex = new double [] {0., 1., 2., 3., 4., 5.};
                var avg = mean(ex);
                var std = sd(ex);
                var exv = Matrix.row(ex);
                exv = exv.sub(avg).div(std);
                System.out.println("standardized:");
                System.out.println(exv);

                // ------------------------------------------------
                // Assessing feature importance with Random Forests
                // -------------------------------------------------

                MathEx.setSeed(19650218);
                //var housing = Read.arff("data/housing.arff");

                var train_data = DataFrame.of(train_dt.toArray(), names);
                System.out.println(train_data);
                var nhs = train_data.drop(0);

                double [] cls = train_data.column("class").toDoubleArray();
                train_data = nhs.merge(IntVector.of("class", roundUp2(cls)));
                System.out.println(train_data);

                System.out.println("Fitting model ...");
                var model = RandomForest.fit(Formula.lhs("class"), train_data);
                var importance = model.importance();
                var shap = model.shap(train_data);
                var fields = java.util.Arrays.copyOf(train_data.names(), 13);

                System.out.println("Importance ...");
                smile.sort.QuickSort.sort(importance, fields);
                for (int i = 0; i < importance.length; i++) {
                    System.out.format("%-15s %12.4f%n", fields[i], importance[i]);
                }

                System.out.println("SHAP -----");
                fields = java.util.Arrays.stream(model.schema().fields()).map(field -> field.name).toArray(String[]::new);
                double [] suba = new double[fields.length];
                System.arraycopy(shap, 0, suba, 0, fields.length);
                smile.sort.QuickSort.sort(suba, fields);
                System.out.println(shap.length);
                Arrays.stream(suba).forEach(num -> System.out.println(num));

                for (int i = 0; i < fields.length; i++) {
                    System.out.format("%-15s %12.4f%n", fields[i], suba[i]);
                }
            } catch( Exception e ) {
                e.printStackTrace();
            }
        }
        System.exit(0);
    }
}
