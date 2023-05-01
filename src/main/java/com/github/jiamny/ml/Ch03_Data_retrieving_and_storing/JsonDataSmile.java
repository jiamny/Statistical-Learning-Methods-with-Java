package com.github.jiamny.ml.Ch03_Data_retrieving_and_storing;

import smile.data.DataFrame;
import smile.data.type.DataTypes;
import smile.data.type.StructField;
import smile.data.type.StructType;
import smile.io.JSON;
import smile.util.Paths;

public class JsonDataSmile {

    public static void main(String [] args) {
        System.out.println("books single line");
        try {
            JSON json = new JSON();
            DataFrame df = json.read("./data/books.json");

            System.out.println(df);
            System.out.println(df.schema());
            System.out.println("nrow: " + df.nrow()); // 7
            System.out.println("ncol = " + df.ncol()); // 10

            StructType schema = DataTypes.struct(
                    new StructField("series_t", DataTypes.StringType),
                    new StructField("pages_i", DataTypes.IntegerType),
                    new StructField("author", DataTypes.StringType),
                    new StructField("price", DataTypes.DoubleType),
                    new StructField("cat", DataTypes.StringType),
                    new StructField("name", DataTypes.StringType),
                    new StructField("genre_s", DataTypes.StringType),
                    new StructField("sequence_i", DataTypes.IntegerType),
                    new StructField("inStock", DataTypes.BooleanType),
                    new StructField("id", DataTypes.StringType)
            );
            assert schema == df.schema() : "schema is df.schema()";

            assert "Percy Jackson and the Olympians" ==  df.get(0, 0) : "Percy Jackson and the Olympians";
            assert  df.get(0, 1).equals(384) ;
            /*
            assertEquals("Rick Riordan", df.get(0, 2));
            assertEquals(12.5, df.getDouble(0, 3), 1E-7);

            assertNull(df.get(6, 0));
            assertEquals(475, df.get(6, 1));
            assertEquals("Michael McCandless", df.get(6, 2));
            assertEquals(30.5, df.getDouble(6, 3), 1E-7);
             */
        } catch(Exception e) {
            e.printStackTrace();
        }
    }
}
