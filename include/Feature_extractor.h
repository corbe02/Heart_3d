#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include "TrackedMatch.h"
#include "OpticalFlow.h"

class FeatureExtractor {
public:
    FeatureExtractor(double thre);
    static cv::Mat adaptiveHistogramEqualization(const cv::Mat &img);
    void featureDetection(const cv::Mat &prev,const cv::Mat &curr,image_transport::Publisher image_pub_,const cv::Mat &depth);
    cv::Mat extractFeatures(const cv::Mat &img, std::vector<cv::Point2f> &keypoints2f);
private:
    bool first_time_;
    double thres_;
    std::vector<cv::Point2f> points_prev_left_ ;
    std::vector<bool> dynamic_points_prev_left_;
    std::vector<cv::Point2f> points_prev_right_ ;
    std::vector<bool> dynamic_points_prev_right_;
    std::vector<cv::Vec3d> triangulatedPoints3D;
    std::vector<cv::Vec3d> movement_data;
    std::vector<double> movement_x, movement_y, movement_z;
    std::vector<cv::Vec3d> tracked3dpts_previous;
    std::vector<TrackedMatch> tracked_matches;
    OpticalFlow opt_flow;



};

#endif // FEATURE_EXTRACTOR_H
