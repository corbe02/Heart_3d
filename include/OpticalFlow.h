#ifndef OPTICALFLOW_H
#define OPTICALFLOW_H

#include <opencv2/opencv.hpp>

#include "TrackedMatch.h"
#include "data.h"
#include <ros/ros.h>

class OpticalFlow{
public:
    OpticalFlow();
    void computeOpticalFlow(const cv::Mat &prev, cv::Mat &current, double &movement_threshold_,cv::Mat &current_originale,const cv::Mat &depth);
    //void publishTrackedMatch(const TrackedMatch& tm);
    static void OpticalFlowTriangulation(const cv::Mat &prev_l, cv::Mat &current_l,
                              const cv::Mat &prev_r, cv::Mat &current_r,
                              double &movement_threshold_,
                              std::vector<TrackedMatch> &tracked_matches);
    void saveTrackedFeatures(const std::vector<TrackedMatch>& tracked_matches, const std::string& filename);
    
private:
    bool first_time_;
    std::vector<cv::Point2f> points_prev_;
    std::vector<bool> dynamic_points_prev;
    std::vector<TrackedMatch> tracked_matches_;
    ros::Publisher tracked_pub_;

};

#endif // OPTICALFLOW_H