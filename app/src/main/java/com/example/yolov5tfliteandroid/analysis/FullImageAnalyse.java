package com.example.yolov5tfliteandroid.analysis;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.util.Log;
import android.util.Size;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.view.PreviewView;

import com.example.yolov5tfliteandroid.MainActivity;
import com.example.yolov5tfliteandroid.detector.Yolov5TFLiteDetector;
import com.example.yolov5tfliteandroid.utils.ImageProcess;

public class FullImageAnalyse implements ImageAnalysis.Analyzer {

    ImageView boxLabelCanvas;
    PreviewView previewView;
    ImageProcess imageProcess;
    int rotation;
    private Yolov5TFLiteDetector yolov5TFLiteDetector;

    public FullImageAnalyse(Context context, PreviewView previewView, ImageView boxLabelCanvas, int rotation) {
        this.previewView = previewView;
        this.boxLabelCanvas = boxLabelCanvas;
        this.rotation = rotation;
        this.imageProcess = new ImageProcess();
        try{
            this.yolov5TFLiteDetector = new Yolov5TFLiteDetector(
                    "yolov5s-fp16-320-metadata.tflite",
                    "coco_label.txt",
                    false,
                    new Size(320,320),
                    new int[]{1,6300,85});
            this.yolov5TFLiteDetector.initialModel(context);
            Log.i("model", "Success loading model" + this.yolov5TFLiteDetector.getModelFile());
        } catch (Exception e) {
            Log.e("image", "load model error: "+ e.getMessage()+e.toString());
        }
    }

    @Override
    public void analyze(@NonNull ImageProxy image) {
        long start = System.currentTimeMillis();
        int previewHeight = previewView.getHeight();
        int previewWidth = previewView.getWidth();

        byte[][] yuvBytes = new byte[3][];
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        int imageHeight = image.getHeight();
        int imagewWidth = image.getWidth();
//        Log.i("image size", imageHeight + "/" + imagewWidth);

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
//        yolov5TFLiteDetector.detect(imageBitmap);
//        Log.i("image ","has alpha"+imageBitmap.hasAlpha());

//        // 图片适应屏幕fill_start格式的bitmap
        double scale = Math.max(
                previewHeight / (double) (rotation % 180 == 0 ? imagewWidth : imageHeight),
                previewWidth / (double) (rotation % 180 == 0 ? imageHeight : imagewWidth)
        );
        Matrix fullScreenTransform = imageProcess.getTransformationMatrix(
                imagewWidth, imageHeight,
                (int) (scale * imageHeight), (int) (scale * imagewWidth),
                90, false
        );



//        // 适应preview的全尺寸bitmap
        Bitmap fullImageBitmap = Bitmap.createBitmap(imageBitmap, 0, 0, imagewWidth, imageHeight, fullScreenTransform, false);
        boxLabelCanvas.setImageBitmap(fullImageBitmap);

        Log.i("image","image size: "+imagewWidth+"/"+imageHeight+" preview size: "+previewWidth+"/"+previewHeight);

//
//        // 图片适应屏幕fill_start格式的bitmap
//        Bitmap cropImageBitmap = Bitmap.createBitmap(
//                fullImageBitmap, 0, 0,
////                        (int) (scale * imagewWidth), (int) (scale * imageHeight)
//                previewWidth, previewHeight
////                        (rotation % 180 == 0 ? previewHeight : previewWidth),
////                        (rotation % 180 == 0 ? previewWidth : previewHeight)
//        );
//
//        // 模型输入的bitmap
//        Matrix previewToModelTransform =
//                imageProcess.getTransformationMatrix(
//                        cropImageBitmap.getWidth(), cropImageBitmap.getHeight(),
//                        300, 300,
//                        0, false);
//        Bitmap modelInputBitmap = Bitmap.createBitmap(cropImageBitmap, 0, 0,
//                cropImageBitmap.getWidth(), cropImageBitmap.getHeight(),
//                previewToModelTransform, false);
//
//        Matrix modelToPreviewTransform = new Matrix();
//        previewToModelTransform.invert(modelToPreviewTransform);
//
////        ArrayList<CatDogDetector.Recognition> detectRes = catDogDetector.detectCatDog(TensorImage.fromBitmap(modelInputBitmap));
//
//        Bitmap emptyCropSizeBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
//        Canvas cropCanvas = new Canvas(emptyCropSizeBitmap);
        // 边框画笔
//        Paint boxPaint = new Paint();
//        boxPaint.setStrokeWidth(5);
//        boxPaint.setStyle(Paint.Style.STROKE);
//        boxPaint.setColor(Color.RED);
//        // 字体画笔
//        Paint textPain = new Paint();
//        textPain.setTextSize(50);
//        textPain.setColor(Color.RED);
//        textPain.setStyle(Paint.Style.FILL);

//        for (CatDogDetector.Recognition res : detectRes.subList(0, 4)) {
//            RectF location = res.getLocation();
//            String label = res.getLabelName();
//            float confidence = res.getConfidence();
//            if (confidence > 0.5) {
//                modelToPreviewTransform.mapRect(location);
//                cropCanvas.drawRect(location, boxPaint);
//                cropCanvas.drawText(label + ":" + String.format("%.2f", confidence), location.left, location.top, textPain);
//            }
//        }
//        canvasView.setImageBitmap(emptyCropSizeBitmap);
//
//
//        long end = System.currentTimeMillis();
//        long costTime = (end - start);
////                labelView.setText(detectRes.get(0).getLabelName());
//        frameSizeTextView.setText(imageHeight + "x" + imagewWidth);
//        inferenceTimeTextView.setText(Long.toString(costTime) + "ms");
        image.close();
    }
}
