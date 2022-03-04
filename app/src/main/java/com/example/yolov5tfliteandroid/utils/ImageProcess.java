package com.example.yolov5tfliteandroid.utils;

import android.graphics.Bitmap;
import android.util.Size;

import androidx.annotation.Nullable;

import org.tensorflow.lite.DataType;

import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;


public class ImageProcess {

    //
    public static TensorImage resizeImage(Bitmap bitmap, Size size) {
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(size.getHeight(), size.getWidth(), ResizeOp.ResizeMethod.BILINEAR))
                        .build();

        TensorImage tImage = new TensorImage(DataType.UINT8);

        // Analysis code for every frame
        // Preprocess the image
        tImage.load(bitmap);
        tImage = imageProcessor.process(tImage);


        // Create a container for the result and specify that this is a quantized model.
        // Hence, the 'DataType' is defined as UINT8 (8-bit unsigned integer)
        TensorBuffer probabilityBuffer =
                TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.FLOAT32);



        return tImage;
    }
}
