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



void OpticalFlow::computeOpticalFlow(const cv::Mat &prev,cv::Mat &current,double &movement_threshold_,cv::Mat &curr_originale,const cv::Mat &depth)
{
    cv::Mat old_gray, new_gray;

    // --- Parametri intrinseci camera---
    double fx = cameraMatrixLeft.at<double>(0, 0);
    double fy = cameraMatrixLeft.at<double>(1, 1);
    double cx = cameraMatrixLeft.at<double>(0, 2);
    double cy = cameraMatrixLeft.at<double>(1, 2);

    // Converti immagini in grigio
    if (prev.type() != CV_8UC1) cv::cvtColor(prev, old_gray, cv::COLOR_BGR2GRAY);
    else old_gray = prev;
    if (current.type() != CV_8UC1) cv::cvtColor(current, new_gray, cv::COLOR_BGR2GRAY);
    else new_gray = current;


// ------------------------------------------------
// -------- DEBUGGING ----------------------------------
// -------------------------------------------------
    cv::Mat depth_float;
    if (depth.empty()) {
        ROS_WARN_STREAM("Depth input is empty!");
        return;
    }
    depth.convertTo(depth_float, CV_32F);

    // Dimensioni delle immagini
    if ( (depth_float.size() != old_gray.size()) ) {ROS_WARN_STREAM("Size mismatch: depth_size=" << depth_float.size() 
                        << " image_size=" << old_gray.size());
    }

    // Vado a vedere il range di valori depth
    double minD=0, maxD=0;
    cv::minMaxLoc(depth_float, &minD, &maxD);
    ROS_INFO_STREAM("Depth raw range: [" << minD << " , " << maxD << "]  (type=" << depth.type() << ")");

// ------------------------------------------------
// -------------------------------------------------

    // Scale factor per la depth
    float scale_factor = 1.0f; // METRI PER UNITA' NELLA DEPTH 
    // ROI mask
    cv::Mat mask = cv::Mat::zeros(old_gray.size(), CV_8UC1);


    int width = old_gray.cols;
    int height = old_gray.rows;
    int margin_right = width / 12;
    int margin_low = height / 10;
    int roi_width    = width / 3;
    int roi_height   = height /4;
    int x_start      = width - roi_width - margin_right;
    int y_start = (height / 2) - (roi_height / 2)-margin_low;
    //int y_start      = height / 20;
    cv::Rect ROI(x_start, y_start, roi_width, roi_height);
    mask(ROI).setTo(255); 

    cv::Rect fullRect(0, 0, old_gray.cols, old_gray.rows);
    // -------------------- INIZIALIZZAZIONE --------------------
    if (first_time_) {
        std::vector<cv::Point2f> initial_points;

        cv::goodFeaturesToTrack(old_gray, initial_points, 5, 0.0005, 20, mask);
        // - old_gray: immagine in scala di grigi su cui cercare i punti
        // - initial_points: vettore di output che conterrà i punti trovati
        // - 100: numero massimo di punti da rilevare
        // - 0.0005 (qualityLevel): qualità minima dei punti (anche punti meno forti vengono considerati)
        // - 15 (minDistance): distanza minima in pixel tra due punti rilevati
        // - mask: maschera opzionale; punti cercati solo dove mask != 0
        // - Algoritmo usato: Shi-Tomasi corner detector


        // affinamento subpixel (utile per campionare la depth più precisamente)
        if (!initial_points.empty()) {
            cv::cornerSubPix(
                    old_gray,          // immagine in scala di grigi
                    initial_points,    // punti da raffinare (input + output)
                    cv::Size(5,5),     // metà finestra di ricerca (quindi 11x11)
                    cv::Size(-1,-1),   // "zero zone": se lasciata (-1,-1) non esclude nulla
                    cv::TermCriteria(  // criterio di arresto
                        cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                        20,            // max 20 iterazioni
                        0.03           // oppure stop se lo spostamento < 0.03 pixel
                    )
);
        }

        tracked_matches_.clear();
        dynamic_points_prev.clear();

        // inizializzazione TrackedMatch con check depth
        for (size_t i = 0; i < initial_points.size(); ++i) {
            const cv::Point2f &p = initial_points[i];
            int px = std::max(0, std::min((int)std::round(p.x), depth_float.cols - 1));
            int py = std::max(0, std::min((int)std::round(p.y), depth_float.rows - 1));
            float depth_value = depth_float.at<float>(py, px);

            if (!std::isfinite(depth_value)) {
                ROS_WARN_STREAM("Non-finite depth at ("<<px<<","<<py<<") -> skipping point");
                continue;
            }

            // Conversione in sistema metrico
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

        }

        first_time_ = false;
        return;
    }

    // -------------------- Costruzione set punti attivi --------------------
    std::vector<cv::Point2f> prevPoints;
    std::vector<size_t> opticalFlowToMatchIdx;

    // Scorri tutti i tracked_matches_. Per ciascuno che ha is_active == true:
    //aggiungi la sua posizione 2D pt in prevPoints (questi sono i punti d’ingresso per l’optical flow),
    // aggiungi l’indice i in opticalFlowToMatchIdx così da poter risalire al TrackedMatch originale dopo il calcolo del flow. --> id globale della feature
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

 // -------------------- OPTICAL FLOW --------------------
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
    // Itero sulle features attive, controllo che siano state tracciate (status = true)
    // Se non sono state tracciate, le disattivo --> is_active = false
    for (size_t flow_idx = 0; flow_idx < opticalFlowToMatchIdx.size(); ++flow_idx) {
        size_t match_idx = opticalFlowToMatchIdx[flow_idx];
        if (!status[flow_idx]) {
            tracked_matches_[match_idx].is_active = false;
            continue;
        }

        // Estraggo le nuove coordinate 3d del punto
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

        // Controllo se è statico o dinamico e aggiorno il punto in trackedmatch
        double movement = cv::norm(new_pt - prevPoints[flow_idx]);

        TrackedMatch &tm = tracked_matches_[match_idx];
        tm.pt = new_pt;
        tm.position_3d = cv::Point3f(X, Y, Z);
        tm.is_active = true;
        if (movement >= movement_threshold_) tm.dynamic_point = true;
        tm.history.push_back(tm.position_3d);

        // // debug: se tutto vicino a zero logga per indagine
        // if (std::fabs(X) < 1e-9 && std::fabs(Y) < 1e-9 && std::fabs(Z) < 1e-9) {
        //     ROS_WARN_STREAM("Tiny 3D coords for match_idx="<<match_idx
        //                     <<" pixel=("<<px<<","<<py<<") depth_raw="<<depth_value
        //                     <<" -> X="<<X<<" Y="<<Y<<" Z="<<Z);
        // }

        good_old.push_back(prevPoints[flow_idx]);
        good_new.push_back(new_pt);
        dynamic_current.push_back(tm.dynamic_point);
    }

    // Salva punti 
    saveTrackedFeatures(tracked_matches_, "/home/corbe/heart_ws/src/heart_pkg/positions/tracked_features5.txt");



    // //Pubblicazione
    // if (!good_old.empty() && !good_new.empty()) {
    //     Visualizer::drawVoronoi(curr_originale, good_new, cv::Scalar(0, 255, 0)); // disegno il diagramma di Voronoi
        
    //         // Disegna vettori di movimento
    //     for (size_t i = 0; i < good_new.size(); ++i) {
    //         cv::Point2f start = good_old[i];  // posizione precedente
    //         cv::Point2f end   = good_new[i];  // posizione attuale
    //         cv::arrowedLine(curr_originale, start, end, cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.7);
    //         // colore rosso, spessore 2, anti-alias, scala freccia 0.3
    //     }
    // OpticalFlowPose::recoverPose(good_old, good_new, dynamic_current, curr_originale);
    // }
    
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


