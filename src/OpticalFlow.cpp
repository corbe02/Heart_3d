#include "OpticalFlow.h"
#include <ros/ros.h>
#include "Visualizer.h"
#include "OpticalFlowPose.h"
#include <image_transport/image_transport.h>
#include "TrackedMatch.h"
#include <geometry_msgs/PointStamped.h>
#include "data.h"

OpticalFlow::OpticalFlow() {}

void OpticalFlow::computeOpticalFlow(const cv::Mat &prev,
                                     cv::Mat &current,
                                     double &movement_threshold_,
                                     cv::Mat &curr_originale,
                                     const cv::Mat &depth)
{
    cv::Mat old_gray, new_gray;

    if (prev.type() != CV_8UC1)
        cv::cvtColor(prev, old_gray, cv::COLOR_BGR2GRAY);
    else
        old_gray = prev;

    if (current.type() != CV_8UC1)
        cv::cvtColor(current, new_gray, cv::COLOR_BGR2GRAY);
    else
        new_gray = current;

    // -------------------- Initialization --------------------
    if (first_time_) {
        std::vector<cv::Point2f> initial_points;

        // Crea maschera nera
        cv::Mat mask = cv::Mat::zeros(old_gray.size(), CV_8UC1);

        // ROI in alto a destra 
        int width = old_gray.cols;
        int height = old_gray.rows;
        int margin_right = width / 10;  // margine che lasci vuoto a destra
        int roi_width    = width / 3;   // quanto è largo il rettangolo
        int roi_height   = height / 7;  // quanto è alto
        int x_start      = width - roi_width - margin_right;
        int y_start      = height / 20; // così non parte proprio dall'angolo in alto
        cv::Rect ROI(x_start, y_start, roi_width, roi_height);

        // Riempi la ROI a 
        mask(ROI).setTo(255);

        // Cerca features solo nella ROI
        cv::goodFeaturesToTrack(
            old_gray,
            initial_points,
            30,      // maxCorners
            0.0005,    // qualityLevel
            15,      // minDistance
            mask     // <---- maschera qui
        );

        tracked_matches_.clear();
        dynamic_points_prev.clear();

        for (const auto &p : initial_points) {
            float z = depth.at<float>(static_cast<int>(p.y), static_cast<int>(p.x));
            TrackedMatch tm;
            tm.pt = p;
            tm.position_3d = cv::Point3f(p.x, p.y, z);
            tm.is_active = true;
            tm.dynamic_point = false;
            tm.history.push_back(tm.position_3d);
            tracked_matches_.push_back(tm);
            dynamic_points_prev.push_back(false);
        }

        first_time_ = false;
        return;
    }


    // -------------------- Build active set --------------------
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
        first_time_ = true; // force reinit next frame
        return;
    }

    // -------------------- Optical flow --------------------
    std::vector<cv::Point2f> nextPoints;
    std::vector<uchar> status;
    std::vector<float> err;

    cv::Size winSize(21, 21);     // finestra di ricerca per ogni livello
    int maxLevel = 3;             // numero di livelli di piramide
    cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 
                            30, 0.01);  // max 30 iterazioni o precisione 0.01

    cv::calcOpticalFlowPyrLK(old_gray, new_gray,
                            prevPoints, nextPoints,
                            status, err,
                            winSize,
                            maxLevel,
                            criteria,
                            0,    // flags
                            0.001 // soglia minima eigenvalue (se >0 scarta punti deboli)
    );

    std::vector<cv::Point2f> good_old;
    std::vector<cv::Point2f> good_new;
    std::vector<bool> dynamic_current;

    // -------------------- Update tracked matches --------------------
    for (size_t flow_idx = 0; flow_idx < opticalFlowToMatchIdx.size(); ++flow_idx) {
        size_t match_idx = opticalFlowToMatchIdx[flow_idx];

        if (!status[flow_idx]) {
            tracked_matches_[match_idx].is_active = false;
            continue;
        }

        cv::Point2f new_pt = nextPoints[flow_idx];
        float z = depth.at<float>(static_cast<int>(new_pt.y), static_cast<int>(new_pt.x));

        double movement = cv::norm(new_pt - prevPoints[flow_idx]);

        TrackedMatch &tm = tracked_matches_[match_idx];
        tm.pt = new_pt;
        tm.position_3d = cv::Point3f(new_pt.x, new_pt.y, z);
        tm.is_active = true;

        // prima: tm.dynamic_point = (movement >= movement_threshold_);
        if (movement >= movement_threshold_)
            tm.dynamic_point = true;  // resta true per sempre se è superata almeno una volta

        tm.history.push_back(tm.position_3d);

        // for pose recovery
        good_old.push_back(prevPoints[flow_idx]);
        good_new.push_back(new_pt);
        dynamic_current.push_back(tm.dynamic_point); // ok, ma qui leggi dal campo persistente
        // // movement magnitude
        // double movement = cv::norm(new_pt - prevPoints[flow_idx]);

        // // update TrackedMatch
        // TrackedMatch &tm = tracked_matches_[match_idx];
        // tm.pt = new_pt;
        // tm.position_3d = cv::Point3f(new_pt.x, new_pt.y, z);
        // tm.is_active = true;
        // tm.dynamic_point = (movement >= movement_threshold_);
        // tm.history.push_back(tm.position_3d);

        // //publishTrackedMatch(tm);

        // // for pose recovery / visualization
        // good_old.push_back(prevPoints[flow_idx]);
        // good_new.push_back(new_pt);
        // dynamic_current.push_back(tm.dynamic_point);
    }

    // -------------------- Recover camera pose --------------------
    if (!good_old.empty() && !good_new.empty()) {
        OpticalFlowPose::recoverPose(good_old, good_new, dynamic_current, curr_originale);
    }

    // keep last frame points for reinitialization if needed
    points_prev_ = good_new;
    dynamic_points_prev = dynamic_current;
}




// void OpticalFlow::publishTrackedMatch(const TrackedMatch& tm) {
//     if (!tm.is_active) return;  // pubblico solo quelli attivi

//     geometry_msgs::PointStamped msg;
//     msg.header.stamp = ros::Time::now();
//     msg.header.frame_id = "camera_link";  // metti il frame giusto della tua camera

//     msg.point.x = tm.position_3d.x;
//     msg.point.y = tm.position_3d.y;
//     msg.point.z = tm.position_3d.z;

//     tracked_pub_.publish(msg);
// }


// void OpticalFlow::computeOpticalFlow(const cv::Mat &prev, cv::Mat &current, double &movement_threshold_, cv::Mat &curr_originale,const cv::Mat &depth)
// {

//     cv::Mat old_gray;
//     cv::Mat new_gray;

//     if (prev.type() != CV_8UC1)
//     {
//         cv::cvtColor(prev, old_gray, cv::COLOR_BGR2GRAY);
//     }
//     else
//     {
//         old_gray = prev; // If it's already grayscale
//     }

//     if (current.type() != CV_8UC1)
//     {
//         cv::cvtColor(current, new_gray, cv::COLOR_BGR2GRAY);
//     }
//     else
//     {
//         new_gray = current; // If it's already grayscale
//     }


//     std::vector<cv::Point2f> points_current; //vettore dinamico che conterrà un numero variabile di punti 2D (cv::Point2f)

//     // Extract good features to track
//     if(first_time_)
//     {
//         //cv::goodFeaturesToTrack(old_gray, points_prev_, 10000, 0.01, 5,mask); //input image, output array, max_points, quality level, min distance, mask
//         cv::goodFeaturesToTrack(old_gray, points_prev_, 2500, 0.001, 50);
//         dynamic_points_prev = std::vector<bool>(points_prev_.size(), false);
//         first_time_ = false;
//         ROS_INFO_STREAM("fatto gftt: " << dynamic_points_prev.size());
//     }

//     std::vector<uchar> status; //vettore che contiene valori di tipo unsigned char (numeri interi senza segno, che varia da 0 a 255)
//     std::vector<float> err;

//     if(points_prev_.empty()) {
//     ROS_WARN("No points to track. Recomputing features.");
//     cv::goodFeaturesToTrack(old_gray, points_prev_, 2500, 0.001, 50);
//     dynamic_points_prev = std::vector<bool>(points_prev_.size(), false);

//     // Se ancora vuoto, esci dalla funzione
//     if(points_prev_.empty()) return;
//     }
//     cv::calcOpticalFlowPyrLK(old_gray, new_gray, points_prev_, points_current,status,err); //immagine precedente, immagine successiva, punti precedenti, punti successivi
//     // points prev sono solo i punti correttamente tracciati nel frame precedente 


//     //points_current contiene le nuove posizioni delle features

//     std::vector<cv::Point2f> good_old; //conterrà i punti del frame precedente che sono stati correttamente mappati in quello corrente 
//     std::vector<cv::Point2f> good_new; //punti mappati nel frame corrente 

//     std::vector<bool> dynamic_current;
//     for (size_t i = 0; i < status.size(); ++i) {
//         if (status[i]) 
//         {  // Se il punto è stato tracciato correttamente
//             good_old.push_back(points_prev_[i]);  //punti precedenti mappati nel frame corrente 
//             good_new.push_back(points_current[i]); //punti mappati nel frame corrente
//             //tracked_matches[i].pt = points_current[i];
//             dynamic_current.push_back(dynamic_points_prev[i]);
//         }
//         else
//         {
//             //tracked_matches[i].is_active = false;
//         }
//     }

//     //Dynamic current contiene i punti che sono stati correttamente tracciati e che erano dinamici nel frame precedente


//     //Features statiche o dinamiche 
//     for (size_t i = 0; i < good_old.size(); ++i) 
//     {
//         double movement = cv::norm(good_new[i]-good_old[i]);
//         if(movement >= movement_threshold_) //the feature is dynamic
//             dynamic_current[i] = true; //aggiungo eventuali altri punti considerati dinamici, ma quelli dinamici non possono tornare statici
//         //else //the feature is static
//             //dynamic[i] = false;
//     }
 
//     //Visualizer::drawDelaunay(current, good_new, cv::Scalar(255, 0, 0)); //disegno i triangoli di Delaunay 
//     //Visualizer::drawVoronoi(current, good_new, cv::Scalar(0, 255, 0)); // disegno il diagramma di Voronoi

//     OpticalFlowPose::recoverPose(good_old, good_new,dynamic_current, curr_originale);
//     points_prev_ = good_new;

//     if (dynamic_points_prev.size() != dynamic_current.size()) 
//     {
//         dynamic_points_prev.resize(dynamic_current.size(), false);  // Inizializza con valore false (statico)
//     }
//     dynamic_points_prev = dynamic_current;

// }


// void OpticalFlow::OpticalFlowTriangulation(const cv::Mat &prev_l, cv::Mat &current_l,
//                                            const cv::Mat &prev_r, cv::Mat &current_r,
//                                            double &movement_threshold_,
//                                            std::vector<TrackedMatch> &tracked_matches) {
//     // 1. Extract coordinates of active features
//     std::vector<cv::Point2f> prevPointsLeft, prevPointsRight;

//     // Keep mapping between optical flow index and tracked_matches index
//     std::vector<size_t> opticalFlowToMatchIdx;

//     //Per fare l'optical flow, devo utilizzare solo le features attive (che ho tracciato nel frame precedente)
//     //Estraggo quindi le features attive e le metto in prevPointsLeft e prevPointsRight
//     //in opticalFlowToMatchIdx metto gli indici delle features attive (ex: 2,7,13...)
//     // for (size_t i = 0; i < tracked_matches.size(); ++i) {
//     //     if (tracked_matches[i].is_active) {
//     //         // Retrieve left and right 2D points from your TrackedMatch structure
//     //         cv::Point2f pt_left = tracked_matches[i].pt_left;   
//     //         cv::Point2f pt_right = tracked_matches[i].pt_right; 

//     //         prevPointsLeft.push_back(pt_left);
//     //         prevPointsRight.push_back(pt_right);
//     //         opticalFlowToMatchIdx.push_back(i);
//     //     }
//     // }

//     if (prevPointsLeft.empty() || prevPointsRight.empty())
//         return; // No active points to track

//     // -------------------------------------------- OPTICAL FLOW ----------------------------------------
//     // 2. Perform optical flow on active points only
//     std::vector<cv::Point2f> nextPointsLeft, nextPointsRight;
//     std::vector<uchar> statusLeft, statusRight;
//     std::vector<float> errLeft, errRight;

//     cv::calcOpticalFlowPyrLK(prev_l, current_l, prevPointsLeft, nextPointsLeft, statusLeft, errLeft);
//     cv::calcOpticalFlowPyrLK(prev_r, current_r, prevPointsRight, nextPointsRight, statusRight, errRight);

//     // 3. Iterate through statuses and map back to tracked_matches
//     // Itero su tutte le features che prima dell'optical flow erano attive (correttamente tracciate nel frame precedente)
//     // Se statusleft o status right sono false, metto is_active a false ---> feature non tracciata 
//     // Per trovare la posizione di quella feature nel vettore generale, utilizzo l'indice che ho salvato in opticalFlowToMatchIdx
//     for (size_t flow_idx = 0; flow_idx < opticalFlowToMatchIdx.size(); ++flow_idx) {
//         size_t match_idx = opticalFlowToMatchIdx[flow_idx];

//         bool left_ok = statusLeft[flow_idx];
//         bool right_ok = statusRight[flow_idx];

//         if (!left_ok || !right_ok) {
//             tracked_matches[match_idx].is_active = false;
//             continue;
//         }
            
//         // Check movement thresholds
//         float dx_left = nextPointsLeft[flow_idx].x - prevPointsLeft[flow_idx].x;
//         float dy_left = nextPointsLeft[flow_idx].y - prevPointsLeft[flow_idx].y;
//         float movementLeft = std::sqrt(dx_left * dx_left + dy_left * dy_left);

//         float dx_right = nextPointsRight[flow_idx].x - prevPointsRight[flow_idx].x;
//         float dy_right = nextPointsRight[flow_idx].y - prevPointsRight[flow_idx].y;
//         float movementRight = std::sqrt(dx_right * dx_right + dy_right * dy_right);

//         if (movementLeft < movement_threshold_ || movementRight < movement_threshold_) {
//             tracked_matches[match_idx].dynamic_point= false;
//         } else {
//             // Update with new positions
//             tracked_matches[match_idx].pt_left = nextPointsLeft[flow_idx];
//             tracked_matches[match_idx].pt_right = nextPointsRight[flow_idx];

//             tracked_matches[match_idx].dynamic_point = true;

//             //re-triangulate if needed
//             //tracked_matches[match_idx].position_3d = TriangulatePoint(...);
//         }
//     }
// }
