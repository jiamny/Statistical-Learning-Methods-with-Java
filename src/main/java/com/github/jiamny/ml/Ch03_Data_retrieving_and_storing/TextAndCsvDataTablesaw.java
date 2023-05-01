package com.github.jiamny.ml.Ch03_Data_retrieving_and_storing;

import tech.tablesaw.api.Table;
import tech.tablesaw.io.csv.CsvReadOptions;

import java.time.format.DateTimeFormatter;

public class TextAndCsvDataTablesaw {

    public static void main(String [] args) {
        // ----------------------------------------------
        // Use Tablesaw package
        Table t = Table.read().file("./data/data_cleaned.csv");
        t.summarize(t.column(0));

        // create an options object with a builder
        CsvReadOptions.Builder builder =
                CsvReadOptions.builder("./data/data_cleaned.csv")
                        .separator('\t')							// table is tab-delimited
                        .header(false);								// no header
                        //.dateFormat(DateTimeFormatter.ISO_LOCAL_DATE)

        CsvReadOptions options = builder.build();

        Table t1 = Table.read().usingOptions(options);
        System.out.println(t1.first(3));
    }
}
