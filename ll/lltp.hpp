#pragma once

// define macro before include this file
// #define LL_USE_CV

#ifdef LL_USE_CV

#pragma message("ll::tp enabled [opencv]")

#include <opencv2/opencv.hpp>

#define LL_IMSHOW(img) \
    cv::imshow(#img, img)

namespace ll{namespace tp{

void plot_line(cv::Mat& img, cv::Point2f p0, cv::Point2f p1, const cv::Scalar& color){
    // simple y=kx+b
    //todo: handle vertical line
    float k =   (p1.y-p0.y)/(p1.x-p0.x);
    float b =   -k*p0.x+p0.y;

    cv::Point lft(0, 0), rt(img.cols, 0);
    lft.y   =   b;
    rt.y    =   k*rt.x+b;
    cv::line(img, lft, rt, color);
};

}}

#endif
