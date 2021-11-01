// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <sys/system_properties.h>
#include <jni.h>
#include <float.h>
#include <string>
#include <vector>
// ncnn
#include "benchmark.h"
#include "c_api.h"
#include "cpu.h"
#include "datareader.h"
#include "net.h"
#include "gpu.h"
#include <map>
#include <android/log.h>
#include "model_head.h"
using namespace std;
using std::vector;

// YOLOX use the same focus in yolov5
class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob, const ncnn::Option &opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float *ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float *outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};
DEFINE_LAYER_CREATOR(YoloV5Focus)

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char *format, void *p) const { return 0; }
    virtual size_t read(void *buf, size_t size) const
    {
        memset(buf, 0, size);
        return size;
    }
};

class BenchmarkNet : public ncnn::Net
{

public:
    string model_name = "";
    int run(int loops, double &time_min, double &time_max, double &time_avg)
    {
        time_min = DBL_MAX;
        time_max = -DBL_MAX;
        time_avg = 0;
        // resolve input shape
        ncnn::Mat in;
        {
            for (int i = 0; i < (int)layers.size(); i++)
            {
                const ncnn::Layer *layer = layers[i];
                if (layer->type != "Input")
                    continue;
                if (blobs[layer->tops[0]].name != "data")
                    continue;
                const ncnn::Mat &shape = layer->top_shapes[0];
                in.create(shape.w, shape.h, shape.c);
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn: current shape.w => ", "%d", shape.w, '%d', shape.h, "%d", shape.c);
                break;
            }
            if (in.empty())
                return -1;
        }
        in.fill(0.01f);
        ncnn::Mat out;
        // warm up
        const int g_warmup_loop_count = 4; // FIXME hardcode
        for (int i = 0; i < g_warmup_loop_count; i++)
        {
            ncnn::Extractor ex = create_extractor();
            ex.input("data", in);
            string model_small_name = "";
            if (model_name.find("picodet") != string::npos)
            {
                model_small_name = "picodet";
            }
            if (model_name.find("nanodet") != string::npos)
            {
                model_small_name = "nanodet";
            }
            if (model_name.find("ppyolo") != string::npos)
            {
                model_small_name = "ppyolo";
            }
            if (model_name.find("yolox") != string::npos)
            {
                model_small_name = "yolox";
            }
            if (model_name.find("yolov3-") != string::npos)
            {
                model_small_name = "yolov3-";
            }
            if (model_name.find("yolov4-") != string::npos)
            {
                model_small_name = "yolov4-";
            }
            if (model_name.find("yolov5n-") != string::npos)
            {
                model_small_name = "yolov5n-";
            }
            if (model_name.find("yolov5s-") != string::npos)
            {
                model_small_name = "yolov5s-";
            }
            __android_log_print(ANDROID_LOG_DEBUG, "current: ", "model_small_name: %s", model_small_name.data());
            vector<string> heads = modelHead[model_small_name];
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn: current head size =>", "%d", heads.size());
            for (int i = 0; i < heads.size(); i++)
            {
                string item_head = heads[i];
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn: current head name =>", "model name: %s", item_head.data());
                ex.extract(item_head.data(), out);
            }
        }

        for (int i = 0; i < loops; i++)
        {
            double start = ncnn::get_current_time();
            {
                ncnn::Extractor ex = create_extractor();
                ex.input("data", in);
                string model_small_name = "";
                if (model_name.find("picodet") != string::npos)
                {
                    model_small_name = "picodet";
                }
                if (model_name.find("nanodet") != string::npos)
                {
                    model_small_name = "nanodet";
                }
                if (model_name.find("ppyolo") != string::npos)
                {
                    model_small_name = "ppyolo";
                }
                if (model_name.find("yolox") != string::npos)
                {
                    model_small_name = "yolox";
                }
                if (model_name.find("yolov3-") != string::npos)
                {
                    model_small_name = "yolov3-";
                }
                if (model_name.find("yolov4-") != string::npos)
                {
                    model_small_name = "yolov4-";
                }
                if (model_name.find("yolov5n-") != string::npos)
                {
                    model_small_name = "yolov5n-";
                }
                if (model_name.find("yolov5s-") != string::npos)
                {
                    model_small_name = "yolov5s-";
                }
                vector<string> heads = modelHead[model_small_name];
                for (int i = 0; i < heads.size(); i++)
                {
                    string item_head = heads[i];
                    ex.extract(item_head.data(), out);
                }
            }
            double end = ncnn::get_current_time();
            double time = end - start;
            time_min = std::min(time_min, time);
            time_max = std::max(time_max, time);
            time_avg += time;
        }
        time_avg /= loops;
        return 0;
    }
};

extern "C"
{

    static jclass objCls = NULL;
    static jmethodID constructortorId;
    static jfieldID retcodeId;
    static jfieldID minId;
    static jfieldID maxId;
    static jfieldID avgId;

    JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved)
    {
        __android_log_print(ANDROID_LOG_DEBUG, "current BenchmarkNcnn", "JNI_OnLoad");
        ncnn::create_gpu_instance();
        return JNI_VERSION_1_4;
    }

    JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved)
    {
        __android_log_print(ANDROID_LOG_DEBUG, "current BenchmarkNcnn", "JNI_OnUnload");
        ncnn::destroy_gpu_instance();
    }

    // public native boolean Init();
    JNIEXPORT jboolean JNICALL Java_com_baidu_mobiledetbenchmark_BenchmarkNcnn_Init(JNIEnv *env, jobject thiz)
    {
        jclass localObjCls = env->FindClass("com/baidu/mobiledetbenchmark/BenchmarkNcnn$Obj");
        objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));
        constructortorId = env->GetMethodID(objCls, "<init>", "(Lcom/baidu/mobiledetbenchmark/BenchmarkNcnn;)V");
        retcodeId = env->GetFieldID(objCls, "retcode", "I");
        minId = env->GetFieldID(objCls, "min", "F");
        maxId = env->GetFieldID(objCls, "max", "F");
        avgId = env->GetFieldID(objCls, "avg", "F");

        return JNI_TRUE;
    }

    // public native String GetPlatform();
    JNIEXPORT jstring JNICALL Java_com_baidu_mobiledetbenchmark_BenchmarkNcnn_GetPlatform(JNIEnv *env, jobject thiz)
    {
        char platform[PROP_VALUE_MAX + 1];
        __system_property_get("ro.board.platform", platform);

        return env->NewStringUTF(platform);
    }

    // public native String GetNcnnVersion();
    JNIEXPORT jstring JNICALL Java_com_baidu_mobiledetbenchmark_BenchmarkNcnn_GetNcnnVersion(JNIEnv *env, jobject thiz)
    {
        return env->NewStringUTF(ncnn_version());
    }

    JNIEXPORT jobject JNICALL Java_com_baidu_mobiledetbenchmark_BenchmarkNcnn_Run(JNIEnv *env, jobject thiz, jobject assetManager, jint threads, jint powersave,
                                                                                  jboolean mempool, jboolean winograd, jboolean sgemm, jboolean pack4, jboolean bf16s,
                                                                                  jboolean gpu, jboolean gpufp16p, jboolean gpufp16s, jboolean gpufp16a, jboolean gpupack8,
                                                                                  jint model, jint loops)
    {
        __android_log_print(ANDROID_LOG_DEBUG, "BenchmarkNcnn", "threads=%d powersave=%d mempool=%d winograd=%d sgemm=%d pack4=%d bf16s=%d gpu=%d gpufp16p=%d gpufp16s=%d gpufp16a=%d gpupack8=%d model=%d loops=%d", threads, powersave, mempool, winograd, sgemm, pack4, bf16s, gpu, gpufp16p, gpufp16s, gpufp16a, gpupack8, model, loops);

        if (gpu == JNI_TRUE && ncnn::get_gpu_count() == 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "current use GPU", "GPU using...");
            jobject jObj = env->NewObject(objCls, constructortorId, thiz);
            env->SetIntField(jObj, retcodeId, 1);
            return jObj;
        }

        if (model < 0 || model >= sizeof(g_models) / sizeof(const char *))
        {
            // unknown model
            jobject jObj = env->NewObject(objCls, constructortorId, thiz);
            env->SetIntField(jObj, retcodeId, 2);
            return jObj;
        }
        __android_log_print(ANDROID_LOG_DEBUG, "current use mode", "%s", g_models[model]);

        ncnn::UnlockedPoolAllocator *blob_pool_allocator = 0;
        ncnn::UnlockedPoolAllocator *workspace_pool_allocator = 0;

        ncnn::VulkanDevice *vkdev = 0;
        ncnn::VkBlobAllocator *blob_vkallocator = 0;
        ncnn::VkStagingAllocator *staging_vkallocator = 0;

        // prepare opt
        ncnn::Option opt;
        opt.lightmode = true;
        opt.num_threads = threads;

        if (mempool)
        {
            blob_pool_allocator = new ncnn::UnlockedPoolAllocator;
            workspace_pool_allocator = new ncnn::UnlockedPoolAllocator;

            opt.blob_allocator = blob_pool_allocator;
            opt.workspace_allocator = workspace_pool_allocator;
        }

        if (gpu)
        {
            const int gpu_device = 0; // FIXME hardcode
            vkdev = ncnn::get_gpu_device(0);

            blob_vkallocator = new ncnn::VkBlobAllocator(vkdev);
            staging_vkallocator = new ncnn::VkStagingAllocator(vkdev);

            opt.blob_vkallocator = blob_vkallocator;
            opt.workspace_vkallocator = blob_vkallocator;
            opt.staging_vkallocator = staging_vkallocator;
        }

        opt.use_winograd_convolution = winograd;
        opt.use_sgemm_convolution = sgemm;

        opt.use_vulkan_compute = gpu;

        opt.use_fp16_packed = gpufp16p;
        opt.use_fp16_storage = gpufp16s;
        opt.use_fp16_arithmetic = gpufp16a;
        opt.use_int8_storage = true;
        opt.use_int8_arithmetic = false;

        opt.use_shader_pack8 = gpupack8;

        opt.use_bf16_storage = bf16s;

        ncnn::set_cpu_powersave(powersave);

        // load model
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);

        BenchmarkNet net;
        net.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
        net.opt = opt;
        if (gpu)
        {
            net.set_vulkan_device(vkdev);
        }
        std::string param_path = g_models[model] + std::string(".param");
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn: current use model", "%s", param_path.data());
        net.model_name = g_models[model];
        net.load_param(mgr, param_path.c_str());
        DataReaderFromEmpty dr;
        net.load_model(dr);
        double time_min;
        double time_max;
        double time_avg;
        int rr = net.run(loops, time_min, time_max, time_avg);

        __android_log_print(ANDROID_LOG_DEBUG, "current use loops", "%d", loops);

        delete blob_pool_allocator;
        delete workspace_pool_allocator;

        delete blob_vkallocator;
        delete staging_vkallocator;

        if (rr != 0)
        {
            // runtime error
            jobject jObj = env->NewObject(objCls, constructortorId, thiz);

            env->SetIntField(jObj, retcodeId, 3);

            return jObj;
        }

        // return result
        jobject jObj = env->NewObject(objCls, constructortorId, thiz);

        env->SetIntField(jObj, retcodeId, 0);
        env->SetFloatField(jObj, minId, time_min);
        env->SetFloatField(jObj, maxId, time_max);
        env->SetFloatField(jObj, avgId, time_avg);

        return jObj;
    }
}
