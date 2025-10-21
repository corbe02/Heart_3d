#ifndef OPTICAL_FLOW_POSE_H
#define OPTICAL_FLOW_POSE_H

// ROS and image transport
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

// OpenCV
#include <opencv2/opencv.hpp>

// Message filters
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// Local includes
#include "Feature_extractor.h"
#include "Visualizer.h"
#include "data.h"
#include "OpticalFlow.h"
#include "TrackedMatch.h"

// --- Define Sync Policy BEFORE class ---
typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::Image,
    sensor_msgs::Image,
    sensor_msgs::Image> MySyncPolicy;

typedef message_filters::Synchronizer<MySyncPolicy> Sync;

class OpticalFlowPose {
public:
    OpticalFlowPose(ros::NodeHandle &nh);

    void imageCallback(const sensor_msgs::ImageConstPtr &msg);

    void rgbdCallback(const sensor_msgs::ImageConstPtr& rgb_msg,
                      const sensor_msgs::ImageConstPtr& depth_msg,
                      const sensor_msgs::ImageConstPtr& mask_msg);

    static void PublishRenderedImage(image_transport::Publisher pub, cv::Mat image, std::string encoding, std::string frame_id);
    
    static void recoverPose(const std::vector<cv::Point2f> &good_old,
                            const std::vector<cv::Point2f> &good_new,
                            const std::vector<bool>& dynamic,
                            cv::Mat &current);

    double movement_threshold_;

private:
    // ROS
    ros::NodeHandle private_nh_;
    image_transport::ImageTransport it_;
    static image_transport::Publisher image_pub_;

    // Subscribers (message_filters)
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
    message_filters::Subscriber<sensor_msgs::Image> mask_sub_;

    // Synchronizer
    boost::shared_ptr<Sync> sync_;

    // Internal components
    FeatureExtractor feature_extractor_;
    Visualizer visualizer_;
    OpticalFlow optical_flow_;

    // Tracking data
    cv::Mat prev_img_;
    cv::Mat current_img_;
    cv::Mat depth_img_;
    cv::Mat mask_img_;
    std::vector<cv::Point2f> points_prev_;
    std::vector<bool> dynamic_points_prev;
    bool first_time_;
};

#endif // OPTICAL_FLOW_POSE_H
