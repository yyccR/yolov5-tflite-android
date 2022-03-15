package com.example.yolov5tfliteandroid.detector;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Binder;
import android.os.Build;
import android.util.Log;
import android.util.Size;

import com.example.yolov5tfliteandroid.utils.Recognition;

import org.checkerframework.checker.nullness.Opt;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;


public class Yolov5TFLiteDetector {

    private final Size INPNUT_SIZE = new Size(320, 320);
    private final int[] OUTPUT_SIZE = new int[]{1, 6300, 85};
    private final Boolean IS_INT8 = false;
    private final float DETECT_THRESHOLD = 0.4f;
    private final float IOU_THRESHOLD = 0.25f;
    private final float IOU_CLASS_DUPLICATED_THRESHOLD = 0.7f;
    private final String MODEL_FILE = "yolov5s-fp16-320-metadata.tflite";
//    private final String MODEL_FILE = "yolov5n-fp16-320.tflite";
    private final String LABEL_FILE = "coco_label.txt";

    private Interpreter tflite;
    private List<String> associatedAxisLabels;
    Interpreter.Options options = new Interpreter.Options();


    public String getModelFile() {
        return this.MODEL_FILE;
    }

    public String getLabelFile() {
        return this.LABEL_FILE;
    }

    public Size getInputSize(){return this.INPNUT_SIZE;}
    public int[] getOutputSize(){return this.OUTPUT_SIZE;}

    /**
     * 初始化模型, 可以通过 addNNApiDelegate(), addGPUDelegate()提前加载相应代理
     *
     * @param activity
     */
    public void initialModel(Context activity) {
        // Initialise the model
        try {

            ByteBuffer tfliteModel = FileUtil.loadMappedFile(activity, MODEL_FILE);
            tflite = new Interpreter(tfliteModel, options);
            Log.i("tfliteSupport", "Success reading model: " + MODEL_FILE);

            associatedAxisLabels = FileUtil.loadLabels(activity, LABEL_FILE);
            Log.i("tfliteSupport", "Success reading label: " + LABEL_FILE);

        } catch (IOException e) {
            Log.e("tfliteSupport", "Error reading model or label: ", e);
        }
    }

    /**
     * 检测步骤
     *
     * @param bitmap
     * @return
     */
    public ArrayList<Recognition> detect(Bitmap bitmap) {

        // yolov5s-tflite的输入是:[1, 320, 320,3], 摄像头每一帧图片需要resize,再归一化
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(INPNUT_SIZE.getHeight(), INPNUT_SIZE.getWidth(), ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(0, 255))
                        .build();
        TensorImage yolov5sTfliteInput = new TensorImage(DataType.FLOAT32);
        yolov5sTfliteInput.load(bitmap);
        yolov5sTfliteInput = imageProcessor.process(yolov5sTfliteInput);


        // yolov5s-tflite的输出是:[1, 6300, 85], 可以从v5的GitHub release处找到相关tflite模型, 输出是[0,1], 处理到320.
        TensorBuffer probabilityBuffer =
                TensorBuffer.createFixedSize(OUTPUT_SIZE, DataType.FLOAT32);

        // 推理计算
        if (null != tflite) {
            // 这里tflite默认会加一个batch=1的纬度
            tflite.run(yolov5sTfliteInput.getBuffer(), probabilityBuffer.getBuffer());
        }
        // 输出数据被平铺了出来
        float[] recognitionArray = probabilityBuffer.getFloatArray();
        // 这里将flatten的数组重新解析(xywh,obj,classes).
        ArrayList<Recognition> allRecognitions = new ArrayList<>();
        for (int i = 0; i < OUTPUT_SIZE[1]; i++) {
            int gridStride = i * OUTPUT_SIZE[2];
            // 由于yolov5作者在导出tflite的时候对输出除以了image size, 所以这里需要乘回去
            float x = recognitionArray[0 + gridStride] * INPNUT_SIZE.getWidth();
            float y = recognitionArray[1 + gridStride] * INPNUT_SIZE.getHeight();
            float w = recognitionArray[2 + gridStride] * INPNUT_SIZE.getWidth();
            float h = recognitionArray[3 + gridStride] * INPNUT_SIZE.getHeight();
            int xmin = (int) Math.max(0, x - w / 2.);
            int ymin = (int) Math.max(0, y - h / 2.);
            int xmax = (int) Math.min(INPNUT_SIZE.getWidth(), x + w / 2.);
            int ymax = (int) Math.min(INPNUT_SIZE.getHeight(), y + h / 2.);
            float confidence = recognitionArray[4 + gridStride];

            float[] classScores = Arrays.copyOfRange(recognitionArray, 5 + gridStride, 85 + gridStride);

            int labelId = 0;
            float maxLabelScores = 0.f;
            for (int j = 0; j < classScores.length; j++) {
                if (classScores[j] > maxLabelScores) {
                    maxLabelScores = classScores[j];
                    labelId = j;
                }
            }


            Recognition r = new Recognition(
                    labelId,
                    "",
                    maxLabelScores,
                    confidence,
                    new RectF(xmin, ymin, xmax, ymax));
            allRecognitions.add(
                    r);
        }

        // 非极大抑制输出
        ArrayList<Recognition> nmsRecognitions = nms(allRecognitions);
        // 第二次非极大抑制, 过滤那些同个目标识别到2个以上目标边框为不同类别的
        ArrayList<Recognition> nmsFilterBoxDuplicationRecognitions = nms(nmsRecognitions);

        // 更新label信息
        for(Recognition recognition : nmsFilterBoxDuplicationRecognitions){
            int labelId = recognition.getLabelId();
            String labelName = associatedAxisLabels.get(labelId);
            recognition.setLabelName(labelName);
        }

        return nmsRecognitions;
    }

    /**
     * 非极大抑制
     *
     * @param allRecognitions
     * @return
     */
    protected ArrayList<Recognition> nms(ArrayList<Recognition> allRecognitions) {
        ArrayList<Recognition> nmsRecognitions = new ArrayList<Recognition>();

        // 遍历每个类别, 在每个类别下做nms
        for (int i = 0; i < OUTPUT_SIZE[2]-5; i++) {
            // 这里为每个类别做一个队列, 把labelScore高的排前面
            PriorityQueue<Recognition> pq =
                    new PriorityQueue<Recognition>(
                            6300,
                            new Comparator<Recognition>() {
                                @Override
                                public int compare(final Recognition l, final Recognition r) {
                                    // Intentionally reversed to put high confidence at the head of the queue.
                                    return Float.compare(r.getConfidence(), l.getConfidence());
                                }
                            });

            // 相同类别的过滤出来, 且obj要大于设定的阈值
            for (int j = 0; j < allRecognitions.size(); ++j) {
                if (allRecognitions.get(j).getLabelId() == i && allRecognitions.get(j).getConfidence() > DETECT_THRESHOLD) {
                    pq.add(allRecognitions.get(j));
                }
            }

            // nms循环遍历
            while (pq.size() > 0) {
                // 概率最大的先拿出来
                Recognition[] a = new Recognition[pq.size()];
                Recognition[] detections = pq.toArray(a);
                Recognition max = detections[0];
                nmsRecognitions.add(max);
                pq.clear();

                for (int k = 1; k < detections.length; k++) {
                    Recognition detection = detections[k];
                    if (boxIou(max.getLocation(), detection.getLocation()) < IOU_THRESHOLD) {
                        pq.add(detection);
                    }
                }
            }
        }
        return nmsRecognitions;
    }

    /**
     * 对所有数据不区分类别做非极大抑制
     *
     * @param allRecognitions
     * @return
     */
    protected ArrayList<Recognition> nmsAllClass(ArrayList<Recognition> allRecognitions) {
        ArrayList<Recognition> nmsRecognitions = new ArrayList<Recognition>();

        PriorityQueue<Recognition> pq =
                new PriorityQueue<Recognition>(
                        100,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(final Recognition l, final Recognition r) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(r.getConfidence(), l.getConfidence());
                            }
                        });

        // 相同类别的过滤出来, 且obj要大于设定的阈值
        for (int j = 0; j < allRecognitions.size(); ++j) {
            if (allRecognitions.get(j).getConfidence() > DETECT_THRESHOLD) {
                pq.add(allRecognitions.get(j));
            }
        }

        while (pq.size() > 0) {
            // 概率最大的先拿出来
            Recognition[] a = new Recognition[pq.size()];
            Recognition[] detections = pq.toArray(a);
            Recognition max = detections[0];
            nmsRecognitions.add(max);
            pq.clear();

            for (int k = 1; k < detections.length; k++) {
                Recognition detection = detections[k];
                if (boxIou(max.getLocation(), detection.getLocation()) < IOU_CLASS_DUPLICATED_THRESHOLD) {
                    pq.add(detection);
                }
            }
        }
        return nmsRecognitions;
    }


    protected float boxIou(RectF a, RectF b) {
        float intersection = boxIntersection(a, b);
        float union = boxUnion(a, b);
        if (union <= 0) return 1;
        return intersection / union;
    }

    protected float boxIntersection(RectF a, RectF b) {
        float maxLeft = a.left > b.left ? a.left : b.left;
        float maxTop = a.top > b.top ? a.top : b.top;
        float minRight = a.right < b.right ? a.right : b.right;
        float minBottom = a.bottom < b.bottom ? a.bottom : b.bottom;
        float w = minRight -  maxLeft;
        float h = minBottom - maxTop;

        if (w < 0 || h < 0) return 0;
        float area = w * h;
        return area;
    }

    protected float boxUnion(RectF a, RectF b) {
        float i = boxIntersection(a, b);
        float u = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
        return u;
    }

    /**
     * 添加NNapi代理
     */
    public void addNNApiDelegate() {
        NnApiDelegate nnApiDelegate = null;
        // Initialize interpreter with NNAPI delegate for Android Pie or above
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            nnApiDelegate = new NnApiDelegate();
            options.addDelegate(nnApiDelegate);
        }
    }

    /**
     * 添加GPU代理
     */
    public void addGPUDelegate() {
        GpuDelegate gpuDelegate = new GpuDelegate();
        options.addDelegate(gpuDelegate);
    }

    /**
     * 添加线程数
     * @param thread
     */
    public void addThread(int thread) {
        options.setNumThreads(thread);
    }

}
