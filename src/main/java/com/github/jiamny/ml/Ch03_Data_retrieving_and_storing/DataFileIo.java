package com.github.jiamny.ml.Ch03_Data_retrieving_and_storing;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

class Word {
    private String token="";
    private String pos="";
    public Word(String tk, String ps) {token=tk; pos = ps;}
    public String getToken() {return token;}
    public String getPos() {return pos;}
}

public class DataFileIo {

    public static void main(String [] args) {
        System.out.println(System.getProperty("user.dir"));
        // ------------------------------------------------
        // file input
        // ------------------------------------------------
        try  {
            List<String> lines = new ArrayList<>();

            InputStream is = new FileInputStream("data/text.txt");
            InputStreamReader isReader = new InputStreamReader(is, StandardCharsets.UTF_8);
            BufferedReader reader = new BufferedReader(isReader);

            while (true) {
                String line = reader.readLine();
                if (line == null) {
                    break;
                }
                lines.add(line);
            }
            isReader.close();

            // shortcut to get BufferedReader for a file directly
            Path path = Paths.get("data/text.txt");
            lines.clear();
            reader = Files.newBufferedReader(path, StandardCharsets.UTF_8);
            // read line-by-line
            while (true) {
                String line = reader.readLine();
                if (line == null) {
                    break;
                }
                lines.add(line);
            }
            reader.close();
            System.out.println(lines);

            // use java NIO
            lines.clear();
            lines = Files.readAllLines(path, StandardCharsets.UTF_8);
            System.out.println(lines);

            // ------------------------------------------------
            // Writing ouput data
            // ------------------------------------------------
            PrintWriter writer = new PrintWriter("output.txt", "UTF-8");
            for (String line : lines) {
                String upperCase = line.toUpperCase(Locale.US);
                writer.println(upperCase);
            }

            // java NIO
            Path output = Paths.get("output.txt");
            BufferedWriter bwriter = Files.newBufferedWriter(output, StandardCharsets.UTF_8);
            for (String line : lines) {
                String upperCase = line.toUpperCase(Locale.US);
                bwriter.write(upperCase);
                bwriter.newLine();
            }

        } catch(IOException e) {
            System.out.println(e.getMessage());
        }

        // -------------------------------------------------
        // Streaming API
        // -------------------------------------------------
        Word[] array = { new Word("My", "RPR"), new Word("dog", "NN"),
                new Word("also", "RB"), new Word("likes", "VB"),
                new Word("eating", "VB"), new Word("sausage", "NN"),
                new Word(".", ".") };

        // Now, we can convert this array to a stream using the Arrays.stream utility method:
        Stream<Word> stream = Arrays.stream(array);

        //Streams can be created from collections using the stream method:
        List<Word> list = Arrays.asList(array);
        stream = list.stream();

        // Suppose we want to keep only tokens which are nouns. With the Streams API, we can do it as follows:
        List<String> nouns = list.stream()
                                .filter(w -> "NN".equals(w.getPos())).map(Word::getToken)
                                .collect(Collectors.toList());
        System.out.println(nouns);

        // Alternatively, we may want to check how many unique POS tags there are in the stream.
        // For this, we can use the toSet collector:
        Set<String> pos = list.stream().map(Word::getPos)
                            .collect(Collectors.toSet());
        System.out.println(pos);

        // When dealing with texts, we may sometimes want to join a sequence of strings together:
        String rawSentence = list.stream()
                                .map(Word::getToken)
                                .collect(Collectors.joining(" "));
        System.out.println(rawSentence);

        // Alternatively, we can group words by their POS tag:
        Map<String, List<Word>> groupByPos = list.stream()
                                            .collect(Collectors.groupingBy(Word::getPos));
        System.out.println(groupByPos.get("VB"));
        System.out.println(groupByPos.get("NN"));

        // map from token to word object
        Map<String, Word> tokenToWord = list.stream()
                .collect(Collectors.toMap(Word::getToken, Function.identity()));
        System.out.println(tokenToWord.get("sausage"));

        // find the maximum length across all words in our sentence:
        int maxTokenLength = list.stream()
                            .mapToInt(w -> w.getToken().length())
                            .max().getAsInt();
        System.out.println(maxTokenLength);

        // to create parallel code; for collections, you just need to call the parallelStream method:
        int[] firstLengths = list.parallelStream()
        .filter(w -> w.getToken().length() % 2 == 0)
        .map(Word::getToken).mapToInt(String::length)
        .sequential()
        .sorted()
        .limit(2)
        .toArray();
        System.out.println(Arrays.toString(firstLengths));

        // to represent a text file as a stream of lines using the Files.lines method:
        Path path = Paths.get("data/text.txt");
        try  {
            Stream<String> lines = Files.lines(path, StandardCharsets.UTF_8);
            double average = lines.flatMap(line -> Arrays.stream(line.split(" ")))
                    .map(String::toLowerCase)
                    .mapToInt(String::length)
                    .average().getAsDouble();
            System.out.println("average token length: " + average);
        } catch(IOException e) {
            System.out.println(e.getMessage());
        }
        System.out.println("Done!");
    }
}
