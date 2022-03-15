package com.example.yolov5tfliteandroid.analysis;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.provider.ContactsContract;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.view.PreviewView;

import com.example.yolov5tfliteandroid.detector.Yolov5TFLiteDetector;
import com.example.yolov5tfliteandroid.utils.ImageProcess;
import com.example.yolov5tfliteandroid.utils.Recognition;

import org.tensorflow.lite.support.image.TensorImage;

import java.util.ArrayList;


public class FullScreenAnalyse implements ImageAnalysis.Analyzer {

    ImageView boxLabelCanvas;
    PreviewView previewView;
    int rotation;
    private TextView inferenceTimeTextView;
    private TextView frameSizeTextView;
    ImageProcess imageProcess;
    private Yolov5TFLiteDetector yolov5TFLiteDetector;

    public FullScreenAnalyse(Context context,
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

        long start = System.currentTimeMillis();
        int previewHeight = previewView.getHeight();
        int previewWidth = previewView.getWidth();
        Log.i("image",""+previewWidth+'/'+previewHeight);

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
        Bitmap cropImageBitmap = Bitmap.createBitmap(
                fullImageBitmap, 0, 0,
                previewWidth, previewHeight
        );

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

        Bitmap emptyCropSizeBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        Canvas cropCanvas = new Canvas(emptyCropSizeBitmap);
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
        boxLabelCanvas.setImageBitmap(emptyCropSizeBitmap);

        long end = System.currentTimeMillis();
        long costTime = (end - start);
        frameSizeTextView.setText(previewHeight + "x" + previewWidth);
        inferenceTimeTextView.setText(Long.toString(costTime) + "ms");
        image.close();
    }
}
