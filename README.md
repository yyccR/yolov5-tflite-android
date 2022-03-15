## Deploy yolov5.tflite in android

### 测试效果

<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/yolov5_tflite_android/yolov5_tflite_android1.jpeg" width="378" height="819"/>    <img src="https://raw.githubusercontent.com/yyccR/Pictures/master/yolov5_tflite_android/yolov5_tflite_android2.jpeg" width="378" height="819"/>

### 如何构建

#### 1. 下载 Android studio

  Android studio 下载地址: `https://developer.android.com/studio` 
  
#### 2. git clone 项目构建

- Android studio git clone 本项目地址: `https://github.com/yyccR/yolov5-tflite-android.git`
- 确保根目录下 `build.gradle` 相关依赖库能正常下载
- 在android studio菜单栏`Build`下`Rebuild Project`
- 打开`com.example.yolov5tfliteandroid.MainActivity`, 编译运行安装
