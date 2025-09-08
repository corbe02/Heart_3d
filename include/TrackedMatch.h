
#ifndef TRACKED_MATCH_H
#define TRACKED_MATCH_H

#include <opencv2/opencv.hpp>

// Attualmente sono interessato a capire 3 cose:
// - se il punto è attivo (quindi se sono ancora capace di tracciarlo)
// - se il punto è statico o dinamico 
// E per riconoscere i punti devo assegnare a tutti un id

struct TrackedMatch {
    int id;
    cv::Point2f pt;
    cv::Point3f position_3d;                   
    bool is_active;
    bool dynamic_point;
    std::vector<cv::Point3f> history;         
};

// struct TrackedMatch {
//     int id;
//     //cv::DMatch match;
//     bool is_active;
//     cv::Vec3d position_3d;
//     bool dynamic_point;
//     std::vector<cv::Point3f> history;

//     cv::Point2f pt;
//     cv::Point2f pt_right;

// };



#endif // TRACKED_MATCH_H