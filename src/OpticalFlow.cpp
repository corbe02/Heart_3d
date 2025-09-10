#include "OpticalFlow.h"
#include <ros/ros.h>
#include "Visualizer.h"
#include "OpticalFlowPose.h"
#include <image_transport/image_transport.h>
#include "TrackedMatch.h"
#include <geometry_msgs/PointStamped.h>
#include "data.h"
#include <fstream>
#include <cmath> // per std::isfinite

// Costruttore vuoto della classe OpticalFlow
OpticalFlow::OpticalFlow() {}



void OpticalFlow::computeOpticalFlow(const cv::Mat &prev,
                                     cv::Mat &current,
                                     double &movement_threshold_,
                                     cv::Mat &curr_originale,
                                     const cv::Mat &depth)
{
    cv::Mat old_gray, new_gray;

    // --- Parametri intrinseci camera (restano invariati) ---
    double fx = cameraMatrixLeft.at<double>(0, 0);
    double fy = cameraMatrixLeft.at<double>(1, 1);
    double cx = cameraMatrixLeft.at<double>(0, 2);
    double cy = cameraMatrixLeft.at<double>(1, 2);

    // Converti immagini in gray
    if (prev.type() != CV_8UC1) cv::cvtColor(prev, old_gray, cv::COLOR_BGR2GRAY);
    else old_gray = prev;

    if (current.type() != CV_8UC1) cv::cvtColor(current, new_gray, cv::COLOR_BGR2GRAY);
    else new_gray = current;

    // --- depth -> float coerente ---
    cv::Mat depth_float;
    if (depth.empty()) {
        ROS_WARN_STREAM("Depth input is empty!");
        return;
    }
    if (depth.type() == CV_32F) {
        depth_float = depth;
    } else if (depth.type() == CV_16U) {
        depth.convertTo(depth_float, CV_32F, 1.0f); // eventualmente moltiplica per fattore se necessario
    } else if (depth.type() == CV_8U) {
        depth.convertTo(depth_float, CV_32F, 1.0f / 255.0f);
    } else {
        depth.convertTo(depth_float, CV_32F);
    }

    // debug: dimensioni e range depth
    if ( (depth_float.size() != old_gray.size()) ) {
        ROS_WARN_STREAM("Size mismatch: depth_size=" << depth_float.size() 
                        << " image_size=" << old_gray.size());
        // non return: potremmo ancora campionare con clamp
    }
    double minD=0, maxD=0;
    cv::minMaxLoc(depth_float, &minD, &maxD);
    ROS_INFO_STREAM("Depth raw range: [" << minD << " , " << maxD << "]  (type=" << depth.type() << ")");

    // Scale factor lo decidi tu; qui lo dichiariamo ma non lo forziamo
    float scale_factor = 1.0f; // METRI PER UNITA' NELLA DEPTH (modificalo da te: es. 0.05, 0.01, 1000 ecc)

    // -------------------- Inizializzazione --------------------
    if (first_time_) {
        std::vector<cv::Point2f> initial_points;

        // ROI mask
        cv::Mat mask = cv::Mat::zeros(old_gray.size(), CV_8UC1);
        int width = old_gray.cols;
        int height = old_gray.rows;
        int margin_right = width / 10;
        int roi_width    = width / 3;
        int roi_height   = height / 7;
        int x_start      = width - roi_width - margin_right;
        int y_start      = height / 20;
        cv::Rect ROI(x_start, y_start, roi_width, roi_height);
        mask(ROI).setTo(255);

        cv::goodFeaturesToTrack(old_gray, initial_points, 30, 0.0005, 15, mask);

        // affinamento subpixel (utile per campionare la depth più precisamente)
        if (!initial_points.empty()) {
            cv::cornerSubPix(old_gray, initial_points, cv::Size(5,5), cv::Size(-1,-1),
                             cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS, 20, 0.03));
        }

        tracked_matches_.clear();
        dynamic_points_prev.clear();

        // inizializzazione TrackedMatch con check depth
        size_t debug_print_n = std::min<size_t>(5, initial_points.size());
        for (size_t i = 0; i < initial_points.size(); ++i) {
            const cv::Point2f &p = initial_points[i];
            int px = std::max(0, std::min((int)std::round(p.x), depth_float.cols - 1));
            int py = std::max(0, std::min((int)std::round(p.y), depth_float.rows - 1));
            float depth_value = depth_float.at<float>(py, px);

            if (!std::isfinite(depth_value)) {
                ROS_WARN_STREAM("Non-finite depth at ("<<px<<","<<py<<") -> skipping point");
                continue;
            }

            float Z = depth_value * scale_factor; // tu imposti scale_factor
            float X = (static_cast<float>(p.x) - static_cast<float>(cx)) * Z / static_cast<float>(fx);
            float Y = (static_cast<float>(p.y) - static_cast<float>(cy)) * Z / static_cast<float>(fy);

            TrackedMatch tm;
            tm.pt = p;
            tm.position_3d = cv::Point3f(X, Y, Z);
            tm.is_active = true;
            tm.dynamic_point = false;
            tm.history.push_back(tm.position_3d);
            tracked_matches_.push_back(tm);
            dynamic_points_prev.push_back(false);

            if (i < debug_print_n) {
                ROS_INFO_STREAM("INIT pt["<<i<<"] pixel=("<<px<<","<<py<<") depth_raw="<<depth_value
                                << " Z(m)="<<Z << " X(m)="<<X << " Y(m)="<<Y);
            }
        }

        first_time_ = false;
        return;
    }

    // -------------------- Costruzione set punti attivi --------------------
    std::vector<cv::Point2f> prevPoints;
    std::vector<size_t> opticalFlowToMatchIdx;
    for (size_t i = 0; i < tracked_matches_.size(); ++i) {
        if (tracked_matches_[i].is_active) {
            prevPoints.push_back(tracked_matches_[i].pt);
            opticalFlowToMatchIdx.push_back(i);
        }
    }
    if (prevPoints.empty()) {
        ROS_WARN("No active points to track. Recomputing features.");
        first_time_ = true;
        return;
    }

    // optical flow
    std::vector<cv::Point2f> nextPoints;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(old_gray, new_gray, prevPoints, nextPoints, status, err,
                             cv::Size(21,21), 3,
                             cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS, 30, 0.01),
                             0, 0.001);

    // update tracked matches
    std::vector<cv::Point2f> good_old, good_new;
    std::vector<bool> dynamic_current;
    for (size_t flow_idx = 0; flow_idx < opticalFlowToMatchIdx.size(); ++flow_idx) {
        size_t match_idx = opticalFlowToMatchIdx[flow_idx];
        if (!status[flow_idx]) {
            tracked_matches_[match_idx].is_active = false;
            continue;
        }

        cv::Point2f new_pt = nextPoints[flow_idx];
        int px = std::max(0, std::min((int)std::round(new_pt.x), depth_float.cols - 1));
        int py = std::max(0, std::min((int)std::round(new_pt.y), depth_float.rows - 1));
        float depth_value = depth_float.at<float>(py, px);

        if (!std::isfinite(depth_value)) {
            ROS_WARN_STREAM("Bad depth at ("<<px<<","<<py<<") -- marking inactive");
            tracked_matches_[match_idx].is_active = false;
            continue;
        }

        float Z = depth_value * scale_factor;
        float X = (new_pt.x - cx) * Z / fx;
        float Y = (new_pt.y - cy) * Z / fy;

        double movement = cv::norm(new_pt - prevPoints[flow_idx]);

        TrackedMatch &tm = tracked_matches_[match_idx];
        tm.pt = new_pt;
        tm.position_3d = cv::Point3f(X, Y, Z);
        tm.is_active = true;
        if (movement >= movement_threshold_) tm.dynamic_point = true;
        tm.history.push_back(tm.position_3d);

        // debug: se tutto vicino a zero logga per indagine
        if (std::fabs(X) < 1e-9 && std::fabs(Y) < 1e-9 && std::fabs(Z) < 1e-9) {
            ROS_WARN_STREAM("Tiny 3D coords for match_idx="<<match_idx
                            <<" pixel=("<<px<<","<<py<<") depth_raw="<<depth_value
                            <<" -> X="<<X<<" Y="<<Y<<" Z="<<Z);
        }

        good_old.push_back(prevPoints[flow_idx]);
        good_new.push_back(new_pt);
        dynamic_current.push_back(tm.dynamic_point);
    }

    // Salva punti (opzionale, dipende dalla tua funzione)
    saveTrackedFeatures(tracked_matches_, "/home/corbe/heart_ws/src/heart_pkg/positions/tracked_features.txt");

    // recover pose se necessario
    if (!good_old.empty() && !good_new.empty()) {
        OpticalFlowPose::recoverPose(good_old, good_new, dynamic_current, curr_originale);
    }

    points_prev_ = good_new;
    dynamic_points_prev = dynamic_current;
}








void OpticalFlow::saveTrackedFeatures(
    const std::vector<TrackedMatch>& tracked_matches,
    const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        ROS_ERROR("Cannot open file %s", filename.c_str());
        return;
    }

    // Salva ogni TrackedMatch
    for (size_t i = 0; i < tracked_matches.size(); ++i) {
        const TrackedMatch& tm = tracked_matches[i];

        // Salvo l'ID del punto, se è dinamico o statico, e tutta la storia 3D
        file << "PointID:" << i 
             << " Dynamic:" << tm.dynamic_point 
             << " History:";

        for (const auto& pos : tm.history) {
            file << "(" << pos.x << "," << pos.y << "," << pos.z << ")";
        }

        file << std::endl;
    }

    file.close();
    ROS_INFO("Tracked features saved to %s", filename.c_str());
}


