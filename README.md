# LightWeight模型图像处理测试应用

这是一个使用LightWeight模型进行图像处理的Android应用。该应用允许用户从相册中选择图片，使用LightWeight模型进行处理，并显示推理时间、FPS和RAM占用情况。

## 功能特点

- 从相册选择图片
- 使用ResNet模型处理图像
- 显示推理时间、FPS和RAM占用
- 简洁直观的用户界面

## 项目结构

```
app/
├── src/main/
│   ├── java/com/example/resnetimageprocessor/
│   │   ├── MainActivity.kt - 主活动，处理UI交互
│   │   ├── ImageProcessor.kt - 图像处理逻辑
│   │   └── PerformanceMonitor.kt - 性能监测工具
│   ├── res/
│   │   ├── layout/ - UI布局文件
│   │   ├── values/ - 资源文件
│   │   └── drawable/ - 图像资源
│   └── assets/ - 存放TFLite模型文件
└── build.gradle - 应用级构建配置
```

## 使用说明

1. 启动应用
2. 点击"选择图片"按钮从相册中选择一张图片
3. 应用将自动使用ResNet模型处理图像
4. 处理完成后，将显示处理结果和性能指标

## 技术栈

- Kotlin
- TensorFlow Lite
- Android Jetpack组件
- Kotlin协程

