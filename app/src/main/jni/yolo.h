#ifndef YOLO_H
#define YOLO_H

#include <opencv2/core/core.hpp>

#include <net.h>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
   
};


class Yolo
{
public:
    Yolo();

    int load(const char* modeltype, int target_size,  const float* norm_vals, bool use_gpu = false);

    int load(AAssetManager* mgr, const char* modeltype, int target_size, const float* norm_vals, bool use_gpu = false);

    int detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold = 0.25f, float nms_threshold = 0.45f);

    int draw(cv::Mat& rgb, const std::vector<Object>& objects);

private:

    ncnn::Net yolo;

    int target_size;
    float norm_vals[3];
    int image_w;
    int image_h;
    int in_w;
    int in_h;

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // YOLO_H
