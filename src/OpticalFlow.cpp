#include "OpticalFlow.h"
#include <ros/ros.h>
#include "Visualizer.h"
#include "OpticalFlowPose.h"
#include <image_transport/image_transport.h>
#include "TrackedMatch.h"
#include <geometry_msgs/PointStamped.h>
#include "data.h"
#include <fstream>

// Costruttore vuoto della classe OpticalFlow
OpticalFlow::OpticalFlow() {}

/*
  Funzione principale per calcolare l'optical flow tra due frame consecutivi
  e aggiornare i punti tracciati.
  Parametri:
  - prev: frame precedente (cv::Mat)
  - current: frame corrente (cv::Mat)
  - movement_threshold_: soglia per distinguere punti dinamici da statici
  - curr_originale: copia originale del frame corrente (per visualizzazione)
  - depth: mappa depth corrispondente al frame corrente
*/
void OpticalFlow::computeOpticalFlow(const cv::Mat &prev,
                                     cv::Mat &current,
                                     double &movement_threshold_,
                                     cv::Mat &curr_originale,
                                     const cv::Mat &depth)
{
    cv::Mat old_gray, new_gray;

    // Estraggo i parametri intrinseci della camera sinistra
    double fx = cameraMatrixLeft.at<double>(0, 0);
    double fy = cameraMatrixLeft.at<double>(1, 1);
    double cx = cameraMatrixLeft.at<double>(0, 2);
    double cy = cameraMatrixLeft.at<double>(1, 2);


    // Se le immagini non sono in scala di grigi, convertile
    if (prev.type() != CV_8UC1)
        cv::cvtColor(prev, old_gray, cv::COLOR_BGR2GRAY);
    else
        old_gray = prev;

    if (current.type() != CV_8UC1)
        cv::cvtColor(current, new_gray, cv::COLOR_BGR2GRAY);
    else
        new_gray = current;

    // -------------------- Inizializzazione --------------------
    if (first_time_) {
        std::vector<cv::Point2f> initial_points;

        // Creazione maschera nera (serve per limitare la ricerca delle feature a una ROI)
        cv::Mat mask = cv::Mat::zeros(old_gray.size(), CV_8UC1); // maschera nera della stessa dimensione dell'immagine originale

        // Definizione della ROI (in alto a destra)
        int width = old_gray.cols;
        int height = old_gray.rows;
        int margin_right = width / 10;  // margine destro da lasciare vuoto
        int roi_width    = width / 3;   // larghezza del rettangolo
        int roi_height   = height / 7;  // altezza del rettangolo
        int x_start      = width - roi_width - margin_right;
        int y_start      = height / 20; // leggermente spostato dall'alto
        cv::Rect ROI(x_start, y_start, roi_width, roi_height); //creo rettangolo con quelle coordinate 

        // Riempie la ROI con 255 nella maschera (dove cercare feature)
        mask(ROI).setTo(255); // --> metto bianca la zona in cui permetto di cercare le features 

        // Rilevamento dei punti feature (goodFeaturesToTrack) solo nella ROI
        cv::goodFeaturesToTrack(
            old_gray,
            initial_points,
            30,      // massimo numero di punti
            0.0005,  // qualityLevel
            15,      // distanza minima tra punti
            mask     // maschera per limitare l'area
        );

        // Pulizia delle strutture dati
        tracked_matches_.clear();
        dynamic_points_prev.clear();

        // Creazione dei TrackedMatch per ogni punto rilevato
        for (const auto &p : initial_points) {
            float depth_value = depth.at<float>(static_cast<int>(p.y), static_cast<int>(p.x));
            float scale_factor = 2.0f; // da calibrare empiricamente

            float Z = depth_value * scale_factor;
            float X = (p.x - cx) * Z / fx;
            float Y = (p.y - cy) * Z / fy;

            TrackedMatch tm;
            tm.pt = p;                              // coordinate 2D
            tm.position_3d = cv::Point3f(X, Y, Z);  // coordinate 3D metriche
            tm.is_active = true;
            tm.dynamic_point = false;
            tm.history.push_back(tm.position_3d);   // salva la storia
            tracked_matches_.push_back(tm);
            dynamic_points_prev.push_back(false);
        }

        first_time_ = false; // inizializzazione completata
        return;
    }

    // -------------------- Costruzione del set di punti attivi --------------------
    std::vector<cv::Point2f> prevPoints;
    std::vector<size_t> opticalFlowToMatchIdx;

    // Considera solo i punti attivi
    for (size_t i = 0; i < tracked_matches_.size(); ++i) {
        if (tracked_matches_[i].is_active) {
            prevPoints.push_back(tracked_matches_[i].pt);
            opticalFlowToMatchIdx.push_back(i);
        }
    }

    if (prevPoints.empty()) {
        ROS_WARN("No active points to track. Recomputing features.");
        first_time_ = true; // forzo reinizializzazione al prossimo frame
        return;
    }

    // -------------------- Calcolo dell'optical flow --------------------
    std::vector<cv::Point2f> nextPoints;
    std::vector<uchar> status;
    std::vector<float> err;

    cv::Size winSize(21, 21);  // finestra di ricerca per LK
    int maxLevel = 3;          // livelli di piramide
    cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 
                              30, 0.01);  // max iterazioni o precisione

    // Optical flow con algoritmo di Lucas-Kanade
    cv::calcOpticalFlowPyrLK(old_gray, new_gray,
                             prevPoints, nextPoints,
                             status, err,
                             winSize,
                             maxLevel,
                             criteria,
                             0,    // flags
                             0.001 // soglia minima eigenvalue
    );

    // -------------------- Aggiornamento dei TrackedMatch --------------------
    std::vector<cv::Point2f> good_old;
    std::vector<cv::Point2f> good_new;
    std::vector<bool> dynamic_current;

    for (size_t flow_idx = 0; flow_idx < opticalFlowToMatchIdx.size(); ++flow_idx) {
    size_t match_idx = opticalFlowToMatchIdx[flow_idx];

    // Se il punto non è stato trovato, lo disattiviamo
    if (!status[flow_idx]) {
        tracked_matches_[match_idx].is_active = false;
        continue;
    }

    cv::Point2f new_pt = nextPoints[flow_idx];
    float depth_value = depth.at<float>(static_cast<int>(new_pt.y), static_cast<int>(new_pt.x));
    float scale_factor = 2.0f; // stesso fattore
    float Z = depth_value * scale_factor;
    float X = (new_pt.x - cx) * Z / fx;
    float Y = (new_pt.y - cy) * Z / fy;

    // Calcolo del movimento
    double movement = cv::norm(new_pt - prevPoints[flow_idx]);

    // Corretto: prendo riferimento all'oggetto già esistente
    TrackedMatch &tm = tracked_matches_[match_idx];
    tm.pt = new_pt;
    tm.position_3d = cv::Point3f(X, Y, Z);
    tm.is_active = true;

    // Se il movimento supera la soglia, il punto diventa dinamico
    if (movement >= movement_threshold_)
        tm.dynamic_point = true; 
    tm.history.push_back(tm.position_3d);

    // Preparazione dei punti per il recover della posa
    good_old.push_back(prevPoints[flow_idx]);
    good_new.push_back(new_pt);
    dynamic_current.push_back(tm.dynamic_point);
    }

    saveTrackedFeatures(tracked_matches_, "/home/corbe/heart_ws/src/heart_pkg/positions/tracked_features2.txt");

     // -------------------- Recupero della posa della camera --------------------
    if (!good_old.empty() && !good_new.empty()) {
        OpticalFlowPose::recoverPose(good_old, good_new, dynamic_current, curr_originale);
    }

    // Aggiorno i punti dell'ultimo frame per eventuale reinizializzazione
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


