#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class Contour
{
    public:
    
    std::vector<cv::Point> contour;
    cv::Rect boundingBox;
    cv::RotatedRect rotatedBoundingBox;
    cv::Point2f rotatedBoundingBoxPoints[4];
    double area;
    double angle;

    Contour();
    Contour(std::vector<cv::Point> &contour);
    bool isValid(double minArea, double minRotation, int error);
};