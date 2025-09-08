#include "Feature_extractor.h"
#include "OpticalFlowPose.h"
#include "OpticalFlow.h"
#include "data.h"
#include "TrackedMatch.h"



FeatureExtractor::FeatureExtractor(double thre):first_time_(true), thres_(thre) {
}

cv::Mat FeatureExtractor::adaptiveHistogramEqualization(const cv::Mat &img) {
    //Increase contrast
    cv::Mat lab_image;
    cv::cvtColor(img, lab_image, cv::COLOR_BGR2Lab);

    // Estrazione L channel
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab_image, lab_planes);


    // CLAHE algorithm per L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    cv::Mat dst;
    clahe->apply(lab_planes[0], dst);

    // Merge
    dst.copyTo(lab_planes[0]);
    cv::merge(lab_planes,lab_image);

    // GRAY
    cv::Mat image_clahe_RGB;
    cv::cvtColor(lab_image, image_clahe_RGB, cv::COLOR_Lab2BGR);
    cv::Mat image_clahe_gray;
    cv::cvtColor(image_clahe_RGB,image_clahe_gray , cv::COLOR_BGR2GRAY);

    return image_clahe_gray;

}

void FeatureExtractor::featureDetection(const cv::Mat &pre,const cv::Mat &current,image_transport::Publisher image_pub_,const cv::Mat &depth) {
    //Aumenta il contrasto e converti in scala di grigi
    cv::Mat current_copy = current.clone();
    cv::Mat curr = adaptiveHistogramEqualization(current);
    cv::Mat prev = adaptiveHistogramEqualization(pre);

    
    opt_flow.computeOpticalFlow(prev, curr, thres_,current_copy,depth);
    

}




cv::Mat FeatureExtractor::extractFeatures(const cv::Mat &img, std::vector<cv::Point2f> &keypoints2f)
{
    cv::Mat gray_img;

    if (img.type() != CV_8UC1)
    {
        cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    } 
    else
    {
        gray_img = img; // If it's already grayscale
    }

    cv::Ptr<cv::ORB> features = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    features->detectAndCompute(gray_img, cv::Mat(), keypoints, descriptors);

    cv::KeyPoint::convert(keypoints, keypoints2f);

    return descriptors;
}