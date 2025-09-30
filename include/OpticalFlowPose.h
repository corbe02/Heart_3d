#ifndef OPTICAL_FLOW_POSE_H
#define OPTICAL_FLOW_POSE_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>


#include "Feature_extractor.h"
#include "Visualizer.h"
#include "data.h"


#include "OpticalFlow.h"
#include "TrackedMatch.h"


class OpticalFlowPose {
public:
    OpticalFlowPose(ros::NodeHandle &nh);
    void imageCallback(const sensor_msgs::ImageConstPtr &msg);
    void rgbdCallback(const sensor_msgs::ImageConstPtr& rgb_msg,const sensor_msgs::ImageConstPtr& depth_msg);
    static void PublishRenderedImage(image_transport::Publisher pub, cv::Mat image, std::string encoding, std::string frame_id);
    static void recoverPose(const std::vector<cv::Point2f> &good_old, const std::vector<cv::Point2f> &good_new,const std::vector<bool>& dynamic, cv::Mat &current);
    double movement_threshold_;


private:

    FeatureExtractor feature_extractor_;
    Visualizer visualizer_;
    OpticalFlow optical_flow_;

    //ros::Subscriber image_sub_;
    // Subscribers
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub_;

    // Sync policy
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> Sync;

    boost::shared_ptr<Sync> sync_;
    ros::NodeHandle private_nh_;
    image_transport::ImageTransport it_;
    static image_transport::Publisher image_pub_;
    cv::Mat prev_img_;
    cv::Mat current_img_;
    cv::Mat depth_img_;
    std::vector<cv::Point2f> points_prev_;
    bool first_time_;
    std::vector<bool> dynamic_points_prev; 


};
    


#endif // OPTICAL_FLOW_POSE_H
