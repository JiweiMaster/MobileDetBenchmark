# Mobile Detection Benchmark
This repo is used to test the speed of the mobile terminal models with NCNN

# Requirements
 - NCNN: ncnn-20201218-android-vulkan

# Quick Start

 1. Download [ncnn-20201218-android-vulkan.zip](https://github.com/Tencent/ncnn/releases) or build ncnn for android yourself

 2. Extract ncnn-20201218-android-vulkan.zip into app/src/main/jni or change the ncnn_DIR path to yours in app/src/main/jni/CMakeLists.txt


 3. Open this project with Android Studio, download model param and extract into app/src/assets


 4. Add input shape for `Input` layer,

#### screenshot
<img src="./screenshot.jpg" width="40%" height="40%"/>

# TODO
## TNN, MNN speed supplement, welcome to contribute!

# Refer from
```
git@github.com:nihui/ncnn-android-benchmark.git
```
