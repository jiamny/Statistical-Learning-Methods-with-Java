package com.github.jiamny.ml.utils;

import java.io.File;
import java.io.IOException;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

/**
 * This class converts images to grayscale color.
 */

public class GrayscaleConverter {

    /**
     * Creates a new grayscaled BufferedImage object from the given source image
     * by averaging each pixels RGB value.
     *
     * @param inputImageAbsPath the absolute path of the image file, including its name and extension.
     * @return a BufferedImage object.
     */
    private BufferedImage compute(String inputImageAbsPath) {

        System.out.println("... Converting source image to gray scale.");

        BufferedImage img = null; // image file

        // Read the source image or throw an exception
        try {
            img = ImageIO.read(new File(inputImageAbsPath));
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Get the image width and height dimensions
        int width = img.getWidth();
        int height = img.getHeight();

        // Convert to grayscale by looping over pixels, beginning at top-most left coordinate (0,0)
        for (int y = 0; y < height; y++) { // y = rows
            for (int x = 0; x < width; x++) { // x = columns

                // Get the pixel value at this (x,y) coordinate
                int p = img.getRGB(x, y);

                // Extract the alpha, R, G, B values from pixel p
                int a = (p >> 24) & 0xff; // Shift bits and unsign
                int r = (p >> 16) & 0xff;
                int g = (p >> 8) & 0xff;
                int b = p & 0xff;

                // Calculate average color (grayscale it)
                int avg = (r + g + b) / 3;

                // Replace RGB value with avg
                p = (a << 24) | (avg << 16) | (avg << 8) | avg;
                img.setRGB(x, y, p);
            }
        }
        return img;
    }

    /**
     * Saves the converted grayscale image. This method builds the save path from the provided file name,
     * file extension, and absolute path of the folder that you want to save the image in.
     *
     * @param path          the absolute path of the folder that you would like to save the image inside.
     * @param imageName     the name you would like to save the image with.
     * @param imageFileType the image file extension, without the dot (.) preceding the image file type.
     * @param image         the BufferedImage object returned from the compute method.
     */
    private void saveImage(String path, String imageName, String imageFileType, BufferedImage image) {

        // Save or throw exception
        try {
            System.out.println("... Saving grayscale image to "
                    + path.concat("\\").concat(imageName).concat(".").concat(imageFileType)); // save path displayed to user

            ImageIO.write(image,
                    imageFileType,
                    new File(path.concat("\\").concat(imageName).concat(".").concat(imageFileType)));

        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("... Image saved.");
    }


    // Driver
    public static void main(String args[]) throws IOException {

        /*
         * Tested for .png and .jpg files. Both worked successfully.
         */

        // Test
        System.out.println("Testing GrayscaleConverter.\n");

        String input = "*source images absolute file path including name and extension*";
        String outputPath = "*absolute path to folder where you will save grayscale image in*";
        String outputFileName = "*save image with this name*";
        String outputFileType = "*save image with this file extension (no dot (.) e.g. png or jpg)*";

        GrayscaleConverter gsc = new GrayscaleConverter();
        BufferedImage convertedImage = gsc.compute(input);
        gsc.saveImage(outputPath, outputFileName, outputFileType, convertedImage);

        System.out.println("\nTest complete.");
    }
}