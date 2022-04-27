package com.example.yolov5tfliteandroid.analysis;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.Log;
import android.util.Size;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.view.PreviewView;

import com.example.yolov5tfliteandroid.MainActivity;
import com.example.yolov5tfliteandroid.detector.Yolov5TFLiteDetector;
import com.example.yolov5tfliteandroid.utils.ImageProcess;
import com.example.yolov5tfliteandroid.utils.Recognition;

import java.util.ArrayList;

import io.reactivex.rxjava3.android.schedulers.AndroidSchedulers;
import io.reactivex.rxjava3.core.Observable;
import io.reactivex.rxjava3.core.ObservableEmitter;
import io.reactivex.rxjava3.core.ObservableOnSubscribe;
import io.reactivex.rxjava3.core.Scheduler;
import io.reactivex.rxjava3.schedulers.Schedulers;

public class FullImageAnalyse implements ImageAnalysis.Analyzer {

    public static class Result{

        public Result(long costTime, Bitmap bitmap) {
            this.costTime = costTime;
            this.bitmap = bitmap;
        }
        long costTime;
        Bitmap bitmap;
    }

    ImageView boxLabelCanvas;
    PreviewView previewView;
    int rotation;
    private TextView inferenceTimeTextView;
    private TextView frameSizeTextView;
    ImageProcess imageProcess;
    private Yolov5TFLiteDetector yolov5TFLiteDetector;

    public FullImageAnalyse(Context context,
                            PreviewView previewView,
                            ImageView boxLabelCanvas,
                            int rotation,
                            TextView inferenceTimeTextView,
                            TextView frameSizeTextView,
                            Yolov5TFLiteDetector yolov5TFLiteDetector) {
        this.previewView = previewView;
        this.boxLabelCanvas = boxLabelCanvas;
        this.rotation = rotation;
        this.inferenceTimeTextView = inferenceTimeTextView;
        this.frameSizeTextView = frameSizeTextView;
        this.imageProcess = new ImageProcess();
        this.yolov5TFLiteDetector = yolov5TFLiteDetector;
    }

    @Override
    public void analyze(@NonNull ImageProxy image) {
        int previewHeight = previewView.getHeight();
        int previewWidth = previewView.getWidth();

        // 这里Observable将image analyse的逻辑放到子线程计算, 渲染UI的时候再拿回来对应的数据, 避免前端UI卡顿
        Observable.create( (ObservableEmitter<Result> emitter) -> {
            long start = System.currentTimeMillis();

            byte[][] yuvBytes = new byte[3][];
            ImageProxy.PlaneProxy[] planes = image.getPlanes();
            int imageHeight = image.getHeight();
            int imagewWidth = image.getWidth();

            imageProcess.fillBytes(planes, yuvBytes);
            int yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();

            int[] rgbBytes = new int[imageHeight * imagewWidth];
            imageProcess.YUV420ToARGB8888(
                    yuvBytes[0],
                    yuvBytes[1],
                    yuvBytes[2],
                    imagewWidth,
                    imageHeight,
                    yRowStride,
                    uvRowStride,
                    uvPixelStride,
                    rgbBytes);

            // 原图bitmap
            Bitmap imageBitmap = Bitmap.createBitmap(imagewWidth, imageHeight, Bitmap.Config.ARGB_8888);
            imageBitmap.setPixels(rgbBytes, 0, imagewWidth, 0, 0, imagewWidth, imageHeight);

            // 图片适应屏幕fill_start格式的bitmap
            double scale = Math.max(
                    previewHeight / (double) (rotation % 180 == 0 ? imagewWidth : imageHeight),
                    previewWidth / (double) (rotation % 180 == 0 ? imageHeight : imagewWidth)
            );
            Matrix fullScreenTransform = imageProcess.getTransformationMatrix(
                    imagewWidth, imageHeight,
                    (int) (scale * imageHeight), (int) (scale * imagewWidth),
                    rotation % 180 == 0 ? 90 : 0, false
            );

            // 适应preview的全尺寸bitmap
            Bitmap fullImageBitmap = Bitmap.createBitmap(imageBitmap, 0, 0, imagewWidth, imageHeight, fullScreenTransform, false);
            // 裁剪出跟preview在屏幕上一样大小的bitmap
            Bitmap cropImageBitmap = Bitmap.createBitmap(fullImageBitmap, 0, 0, previewWidth, previewHeight);

            // 模型输入的bitmap
            Matrix previewToModelTransform =
                    imageProcess.getTransformationMatrix(
                            cropImageBitmap.getWidth(), cropImageBitmap.getHeight(),
                            yolov5TFLiteDetector.getInputSize().getWidth(),
                            yolov5TFLiteDetector.getInputSize().getHeight(),
                            0, false);
            Bitmap modelInputBitmap = Bitmap.createBitmap(cropImageBitmap, 0, 0,
                    cropImageBitmap.getWidth(), cropImageBitmap.getHeight(),
                    previewToModelTransform, false);

            Matrix modelToPreviewTransform = new Matrix();
            previewToModelTransform.invert(modelToPreviewTransform);

            ArrayList<Recognition> recognitions = yolov5TFLiteDetector.detect(modelInputBitmap);
//            ArrayList<Recognition> recognitions = yolov5TFLiteDetector.detect(imageBitmap);

            Bitmap emptyCropSizeBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
            Canvas cropCanvas = new Canvas(emptyCropSizeBitmap);
//            Paint white = new Paint();
//            white.setColor(Color.WHITE);
//            white.setStyle(Paint.Style.FILL);
//            cropCanvas.drawRect(new RectF(0,0,previewWidth, previewHeight), white);
            // 边框画笔
            Paint boxPaint = new Paint();
            boxPaint.setStrokeWidth(5);
            boxPaint.setStyle(Paint.Style.STROKE);
            boxPaint.setColor(Color.RED);
            // 字体画笔
            Paint textPain = new Paint();
            textPain.setTextSize(50);
            textPain.setColor(Color.RED);
            textPain.setStyle(Paint.Style.FILL);

            for (Recognition res : recognitions) {
                RectF location = res.getLocation();
                String label = res.getLabelName();
                float confidence = res.getConfidence();
                modelToPreviewTransform.mapRect(location);
                cropCanvas.drawRect(location, boxPaint);
                cropCanvas.drawText(label + ":" + String.format("%.2f", confidence), location.left, location.top, textPain);
            }
            long end = System.currentTimeMillis();
            long costTime = (end - start);
            image.close();
            emitter.onNext(new Result(costTime, emptyCropSizeBitmap));
//            emitter.onNext(new Result(costTime, imageBitmap));

        }).subscribeOn(Schedulers.io()) // 这里定义被观察者,也就是上面代码的线程, 如果没定义就是主线程同步, 非异步
                // 这里就是回到主线程, 观察者接受到emitter发送的数据进行处理
                .observeOn(AndroidSchedulers.mainThread())
                // 这里就是回到主线程处理子线程的回调数据.
                .subscribe((Result result) -> {
                    boxLabelCanvas.setImageBitmap(result.bitmap);
                    frameSizeTextView.setText(previewHeight + "x" + previewWidth);
                    inferenceTimeTextView.setText(Long.toString(result.costTime) + "ms");
                });

    }
}
