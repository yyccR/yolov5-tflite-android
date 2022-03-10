package com.example.yolov5tfliteandroid.detector;
import android.content.Context;
import android.graphics.Bitmap;
import android.os.Binder;
import android.os.Build;
import android.util.Log;

import com.example.yolov5tfliteandroid.utils.Recognition;

import org.checkerframework.checker.nullness.Opt;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;


public class Yolov5TFLiteDetector {

    private Boolean isInt8;
    private float detectThreshold;
    private String modelFile;
    private String labelFile;
    private Interpreter tflite;
    private List<String> associatedAxisLabels;
    Interpreter.Options options;

    public Yolov5TFLiteDetector(String modelFile, String labelFile, Boolean isInt8){
        this.modelFile = modelFile;
        this.labelFile = labelFile;
        this.isInt8 = isInt8;
        this.options = new Interpreter.Options();
        this.detectThreshold = (float) 0.5;
    }

    /**
     * 初始化模型, 可以通过 addNNApiDelegate(), addGPUDelegate()提前加载相应代理
     * @param activity
     */
    public void initialModel(Context activity) {
        // Initialise the model
        try{

            ByteBuffer tfliteModel
                    = FileUtil.loadMappedFile(activity, modelFile);
            tflite = new Interpreter(tfliteModel, options);
            Log.i("tfliteSupport", "Success reading model: " + modelFile);

            associatedAxisLabels = FileUtil.loadLabels(activity, labelFile);
            Log.i("tfliteSupport", "Success reading label: " + labelFile);


        } catch (IOException e){
            Log.e("tfliteSupport", "Error reading model or label: ", e);
        }
    }

    /**
     * 检测步骤
     * @param bitmap
     * @return
     */
    public ArrayList<Recognition> detect(Bitmap bitmap) {

        final ArrayList<Recognition> recognitions = new ArrayList<>();

        // yolov5s-tflite的输入是:[1, 320, 320,3], 摄像头每一帧图片需要resize,再归一化
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(320, 320, ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(0,255))
                        .build();
        TensorImage yolov5sTfliteInput = new TensorImage(DataType.FLOAT32);
        yolov5sTfliteInput.load(bitmap);
        yolov5sTfliteInput = imageProcessor.process(yolov5sTfliteInput);


        // yolov5s-tflite的输出是:[1, 6300, 85], 可以从v5的GitHub release处找到相关tflite模型.
        TensorBuffer probabilityBuffer =
                TensorBuffer.createFixedSize(new int[]{1, 6300, 85}, DataType.FLOAT32);

        // 推理计算
        if(null != tflite) {
            // 这里tflite默认会加一个batch=1的纬度
            tflite.run(yolov5sTfliteInput.getBuffer(), probabilityBuffer.getBuffer());
        }

        // 输出数据需要放大到图片尺度下

        if (null != associatedAxisLabels) {
            // Map of labels and their corresponding probability
            TensorLabel labels = new TensorLabel(associatedAxisLabels,
                    probabilityProcessor.process(probabilityBuffer));

            // Create a map to access the result based on label
            Map<String, Float> floatMap = labels.getMapWithFloatValue();
        }

        return recognitions;
    }


    /**
     * 非极大抑制
     * @param recognitions
     * @return
     */
    private ArrayList<Recognition> nms(ArrayList<Recognition> recognitions){



    }


    /**
     * 添加NNapi代理
     */
    public void addNNApiDelegate(){
        NnApiDelegate nnApiDelegate = null;
        // Initialize interpreter with NNAPI delegate for Android Pie or above
        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            nnApiDelegate = new NnApiDelegate();
            options.addDelegate(nnApiDelegate);
        }
    }

    /**
     * 添加GPU代理
     */
    public void addGPUDelegate(){
        GpuDelegate gpuDelegate = new GpuDelegate();
        options.addDelegate(gpuDelegate);
    }




}
