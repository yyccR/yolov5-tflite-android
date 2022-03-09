package com.example.yolov5tfliteandroid.detector;
import android.content.Context;
import android.util.Log;

import org.checkerframework.checker.nullness.Opt;
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
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.util.List;
import java.util.Map;


public class Yolov5TFLiteDetector {

    private Interpreter tflite;
    private Boolean isInt8;
    Interpreter.Options options;

    public Yolov5TFLiteDetector(Boolean isInt8){
        this.isInt8 = isInt8;
        this.options = new Interpreter.Options();
    }



    public void initialModel(Context activity, TensorImage image) {

        // yolov5s输出大小以及格式, 可以从v5的GitHub release处找到相关tflite模型.
        TensorBuffer probabilityBuffer =
                TensorBuffer.createFixedSize(new int[]{1, 6300, 85}, DataType.FLOAT32);

        // Initialise the model
        try{
            ByteBuffer tfliteModel
                    = FileUtil.loadMappedFile(activity,
                    "mobilenet_v1_1.0_224_quant.tflite");
            options.setUseNNAPI(true);
            options.setNumThreads(1);
            Interpreter tflite = new Interpreter(tfliteModel, options);


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
