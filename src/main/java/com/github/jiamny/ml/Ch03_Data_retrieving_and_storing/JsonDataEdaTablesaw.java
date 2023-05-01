package com.github.jiamny.ml.Ch03_Data_retrieving_and_storing;

import tech.tablesaw.api.Table;
import tech.tablesaw.io.csv.CsvReadOptions;

public class JsonDataEdaTablesaw {

    public static void main(String [] args) {
        try {
            Table t = Table.read().file("./data/data_cleaned.csv");
            t.summarize(t.column(0));

            // create an options object with a builder
            CsvReadOptions.Builder builder =
                    CsvReadOptions.builder("./data/data_cleaned.csv")
                            .separator('\t')							// table is tab-delimited
                            .header(false);								// no header
            //.dateFormat("yyyy.MM.dd");  				// the date format to use.

            CsvReadOptions options = builder.build();

            Table t1 = Table.read().usingOptions(options);
            System.out.println(t1.first(3));

        } catch(Exception e) {
            e.printStackTrace();
        }
    }
}
