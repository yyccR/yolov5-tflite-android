package com.example.yolov5tfliteandroid;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.camera.view.PreviewView;

import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.View;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.Switch;
import android.widget.TextView;
import androidx.camera.lifecycle.ProcessCameraProvider;

import com.example.yolov5tfliteandroid.analysis.FullImageAnalyse;
import com.example.yolov5tfliteandroid.analysis.FullScreenAnalyse;
import com.example.yolov5tfliteandroid.detector.Yolov5TFLiteDetector;
import com.example.yolov5tfliteandroid.utils.CameraProcess;
import com.google.common.util.concurrent.ListenableFuture;

public class MainActivity extends AppCompatActivity {

    private PreviewView cameraPreviewMatch;
    private PreviewView cameraPreviewWrap;
    private ImageView boxLabelCanvas;
    private Switch immersive;
    private TextView inferenceTimeTextView;
    private TextView frameSizeTextView;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private Yolov5TFLiteDetector yolov5TFLiteDetector;

    private CameraProcess cameraProcess = new CameraProcess();

    /**
     * 获取屏幕旋转角度,0表示拍照出来的图片是横屏
     * @return
     */
    protected int getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }

//    @Override
//    public void onWindowFocusChanged(boolean hasFocus) {
//        super.onWindowFocusChanged(hasFocus);
//        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LAYOUT_STABLE | View.SYSTEM_UI_FLAG_FULLSCREEN | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN);
//        getWindow().setStatusBarColor(Color.TRANSPARENT);
//    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

//        Toolbar toolbar = (Toolbar) findViewById(R.id.my_toolbar);
//        setSupportActionBar(toolbar);

        // 打开app的时候隐藏顶部状态栏
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LAYOUT_STABLE | View.SYSTEM_UI_FLAG_FULLSCREEN | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN);
        getWindow().setStatusBarColor(Color.TRANSPARENT);

        // 全屏画面
        cameraPreviewMatch = findViewById(R.id.camera_preview_match);
        cameraPreviewMatch.setScaleType(PreviewView.ScaleType.FILL_START);

        // 全图画面
        cameraPreviewWrap = findViewById(R.id.camera_preview_wrap);
        cameraPreviewWrap.setScaleType(PreviewView.ScaleType.FILL_START);

        // box/label画面
        boxLabelCanvas = findViewById(R.id.box_label_canvas);

        // 沉浸式体验按钮
        immersive = findViewById(R.id.immersive);

        // 实时更新的一些view
        inferenceTimeTextView = findViewById(R.id.inference_time);
        frameSizeTextView = findViewById(R.id.frame_size);
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        // 申请摄像头权限
        if(!cameraProcess.allPermissionsGranted(this)){
            cameraProcess.requestPermissions(this);
        }

        // 获取手机摄像头拍照旋转参数
        int rotation = getWindowManager().getDefaultDisplay().getRotation();
        Log.i("image", "rotation: " + rotation);

        cameraProcess.showCameraSupportSize(MainActivity.this);

        // 加载模型
        try{
            this.yolov5TFLiteDetector = new Yolov5TFLiteDetector();
            this.yolov5TFLiteDetector.initialModel(this);
//            this.yolov5TFLiteDetector.addNNApiDelegate();
//            this.yolov5TFLiteDetector.addGPUDelegate();
            Log.i("model", "Success loading model" + this.yolov5TFLiteDetector.getModelFile());
        } catch (Exception e) {
            Log.e("image", "load model error: "+ e.getMessage()+e.toString());
        }

        // 默认打开时时全图模式
        cameraPreviewMatch.removeAllViews();
        boxLabelCanvas.setImageResource(0);
        FullImageAnalyse fullImageAnalyse = new FullImageAnalyse(
                MainActivity.this,
                cameraPreviewWrap,
                boxLabelCanvas,
                rotation,
                inferenceTimeTextView,
                frameSizeTextView,
                yolov5TFLiteDetector);
        cameraProcess.startCamera(MainActivity.this, fullImageAnalyse, cameraPreviewWrap);

        immersive.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                if(b){
                    // 进入全屏模式
                    cameraPreviewWrap.removeAllViews();
                    FullScreenAnalyse fullScreenAnalyse = new FullScreenAnalyse(MainActivity.this,
                            cameraPreviewMatch,
                            boxLabelCanvas,
                            rotation,
                            inferenceTimeTextView,
                            frameSizeTextView,
                            yolov5TFLiteDetector);
                    cameraProcess.startCamera(MainActivity.this, fullScreenAnalyse, cameraPreviewMatch);

                }else{
                    // 进入全图模式
                    cameraPreviewMatch.removeAllViews();
                    boxLabelCanvas.setImageResource(0);
                    FullImageAnalyse fullImageAnalyse = new FullImageAnalyse(
                            MainActivity.this,
                            cameraPreviewWrap,
                            boxLabelCanvas,
                            rotation,
                            inferenceTimeTextView,
                            frameSizeTextView,
                            yolov5TFLiteDetector);
                    cameraProcess.startCamera(MainActivity.this, fullImageAnalyse, cameraPreviewWrap);
                }
            }
        });


    }
}