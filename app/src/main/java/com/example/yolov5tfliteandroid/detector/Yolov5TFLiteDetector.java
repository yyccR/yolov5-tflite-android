package com.example.yolov5tfliteandroid.detector;
import android.content.Context;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.List;
import java.util.Map;


public class Yolov5TFLiteDetector {

    private Interpreter tflite;

    public void initialModel(Context activity, TensorImage image) {

        // Create a container for the result and specify that this is a quantized model.
        // Hence, the 'DataType' is defined as UINT8 (8-bit unsigned integer)
        TensorBuffer probabilityBuffer =
                TensorBuffer.createFixedSize(new int[]{1, 6300, 85}, DataType.FLOAT32);



        // Initialise the model
        try{
            MappedByteBuffer tfliteModel
                    = FileUtil.loadMappedFile(activity,
                    "mobilenet_v1_1.0_224_quant.tflite");
            Interpreter tflite = new Interpreter(tfliteModel);


        } catch (IOException e){
            Log.e("tfliteSupport", "Error reading model", e);
        }

// Running inference
        if(null != tflite) {
            tflite.run(image.getBuffer(), probabilityBuffer.getBuffer());
        }



        final String ASSOCIATED_AXIS_LABELS = "labels.txt";
        List<String> associatedAxisLabels = null;

        try {
            associatedAxisLabels = FileUtil.loadLabels(activity, ASSOCIATED_AXIS_LABELS);
        } catch (IOException e) {
            Log.e("tfliteSupport", "Error reading label file", e);
        }

        TensorProcessor probabilityProcessor =
                new TensorProcessor.Builder().add(new NormalizeOp(0, 255)).build();

        if (null != associatedAxisLabels) {
            // Map of labels and their corresponding probability
            TensorLabel labels = new TensorLabel(associatedAxisLabels,
                    probabilityProcessor.process(probabilityBuffer));

            // Create a map to access the result based on label
            Map<String, Float> floatMap = labels.getMapWithFloatValue();
        }

    }
}
