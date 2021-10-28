#ifndef __MODEL_HEAD__
#define __MPDEL_HEAD__
#include <string>
#include <vector>
using namespace std;
using std::vector;



map<string, vector<string>> modelHead = {
        {"picodet",
                {
                        {"save_infer_model/scale_0.tmp_1"},
                        {"save_infer_model/scale_4.tmp_1"},
                        {"save_infer_model/scale_1.tmp_1"},
                        {"save_infer_model/scale_5.tmp_1"},
                        {"save_infer_model/scale_2.tmp_1"},
                        {"save_infer_model/scale_6.tmp_1"},
                        {"save_infer_model/scale_3.tmp_1"},
                        {"save_infer_model/scale_7.tmp_1"}
                }
        },
        {
            "nanodet",
                {
                        {"cls_pred_stride_8"},
                        {"dis_pred_stride_8"},
                        {"cls_pred_stride_16"},
                        {"dis_pred_stride_16"},
                        {"cls_pred_stride_32"},
                        {"dis_pred_stride_32"}
                }
        },
        {
                "ppyolo",
                {
                        {"save_infer_model/scale_0.tmp_1"},
                        {"save_infer_model/scale_1.tmp_1"},
                        {"save_infer_model/scale_2.tmp_1"}
                }
        },
        {
            "yolox",
                {
                        {"output"}
                }
        },
        {
                "yolov3-",
                {
                        {"output"}
                }
        },
        {
                "yolov4-",
                {
                        {"output"}
                }
        },
        {
                "yolov5n-",
                {
                        {"output"},
                        {"375"},
                        {"400"}
                }
        },
        {
                "yolov5s-",
                {
                        {"output"},
                        {"375"},
                        {"400"}
                    }
        }
};


const char* g_models[] =
        {
//                "yolov3-tiny-416",
                "yolov4-tiny-416",
                "ppyolo-tiny-650e-320",
                "ppyolo-tiny-650e-416",
                "nanodet-m-320",
                "nanodet-m-416",
                "nanodet-m-1_5x-416",
                "yolox-nano-416",
                "yolox-tiny-416",
                "yolov5n-640",
                "yolov5s-640",
                "picodet-shufflenetv2-1x-416",
                "picodet-mobilenetv3-large-1x-416",
                "picodet-lcnet-1_5x-416",
                "picodet-s-320",
                "picodet-s-416",
                "picodet-m-320",
                "picodet-m-416",
                "picodet-l-320",
                "picodet-l-416",
                "picodet-l-640",
                "picodet-m-416-464channel",
                "test-ppyolo-tiny-650e-320",
                "test-ppyolo-tiny-650e-416",
                "picodet-l-640-lastest",
        };


#endif