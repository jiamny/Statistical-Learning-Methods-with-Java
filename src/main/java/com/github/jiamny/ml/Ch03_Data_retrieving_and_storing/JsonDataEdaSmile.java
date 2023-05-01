package com.github.jiamny.ml.Ch03_Data_retrieving_and_storing;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import com.github.jiamny.ml.utils.RankedPage;

import com.fasterxml.jackson.jr.ob.JSON;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import smile.data.DataFrame;
import smile.io.*;

public class JsonDataEdaSmile {

    public static List<RankedPage> readRankedPages() throws IOException {
        Path path = Paths.get("./data/ranked-pages.json");
        try (Stream<String> lines = Files.lines(path)) {
            return lines.map(line -> parseJson(line)).collect(Collectors.toList());
        }
    }
    public static RankedPage parseJson(String line) {
        RankedPage rpg = null;
        try {
            rpg = JSON.std.beanFrom(RankedPage.class, line);
        } catch (IOException e) {
            //throw Throwables.getRootCause(e);
            e.printStackTrace();
        }
        return rpg;
    }

    public static void main(String [] args) {

        try {
            List<RankedPage> data = null;
            data = readRankedPages();
            double[] dataArray = data.stream()
                    .mapToDouble(RankedPage::getBodyContentLength)
                    .toArray();
            DescriptiveStatistics desc = new DescriptiveStatistics(dataArray);

            System.out.printf("min: %9.1f%n", desc.getMin());
            System.out.printf("p05: %9.1f%n", desc.getPercentile(5));
            System.out.printf("p25: %9.1f%n", desc.getPercentile(25));
            System.out.printf("p50: %9.1f%n", desc.getPercentile(50));
            System.out.printf("p75: %9.1f%n", desc.getPercentile(75));
            System.out.printf("p95: %9.1f%n", desc.getPercentile(95));
            System.out.printf("max: %9.1f%n", desc.getMax());


        } catch(Exception e) {
            e.printStackTrace();
        }
    }
}
