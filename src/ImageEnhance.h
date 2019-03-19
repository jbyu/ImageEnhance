#ifndef _IMAGE_ENHANCE_H_
#define _IMAGE_ENHANCE_H_

#include "opencv2/opencv.hpp"

cv::Mat multi_scale_retinex(const cv::Mat &img, const std::vector<double>& weights, const std::vector<double>& sigmas, int gain, int offset);

cv::Mat ALTM_retinex(const cv::Mat& img, bool LocalAdaptation = false, bool ContrastCorrect = true);

cv::Mat adaptive_logarithmic_mapping(const cv::Mat& img);

#endif//_IMAGE_ENHANCE_H_
