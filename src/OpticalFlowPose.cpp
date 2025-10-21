#include "OpticalFlowPose.h"
#include "TrackedMatch.h"
#include "data.h"

// Dichiarazione del publisher statico per pubblicare immagini elaborate
image_transport::Publisher OpticalFlowPose::image_pub_;

// Costruttore della classe OpticalFlowPose
OpticalFlowPose::OpticalFlowPose(ros::NodeHandle &nh) 
    : it_(nh),                  // Inizializzazione dell'image_transport
      private_nh_("~"),         // NodeHandle privato per leggere parametri ROS
      first_time_(true),        // Flag per la prima immagine
      movement_threshold_(10.0),// Soglia di movimento per il feature extractor
      feature_extractor_(movement_threshold_), // Oggetto per estrazione delle feature
      visualizer_(),            // Oggetto per visualizzazione (non dettagliato qui)
      optical_flow_()           // Oggetto per calcolo optical flow (non dettagliato qui)
{
    // Carica un parametro ROS "threshold", se non presente usa 10.0
    private_nh_.param("threshold", movement_threshold_, 10.0);

    // Subscriber con message_filters per sincronizzare RGB e depth
    rgb_sub_.subscribe(nh, "/video1/image_raw", 10);  // Subscribe al topic RGB
    depth_sub_.subscribe(nh, "/video1/depth", 10);    // Subscribe al topic Depth
    mask_sub_.subscribe(nh, "/video1/mask", 10);    // Subscribe al topic Depth

    //synchronizer with all three
    sync_.reset(new Sync(MySyncPolicy(10), rgb_sub_, depth_sub_, mask_sub_));

    // Register callback that takes 3 parameters
    sync_->registerCallback(boost::bind(&OpticalFlowPose::rgbdCallback, this, _1, _2, _3));

    // Publisher dell'immagine elaborata
    image_pub_ = it_.advertise("/optical_flow/output_video", 1);  
}

// Callback per la ricezione di immagini RGB e Depth e mask sincronizzate
void OpticalFlowPose::rgbdCallback(
    const sensor_msgs::ImageConstPtr& rgb_msg,
    const sensor_msgs::ImageConstPtr& depth_msg,
    const sensor_msgs::ImageConstPtr& mask_msg)
{
    cv_bridge::CvImagePtr cv_rgb, cv_depth, cv_mask; // Convertitori ROS -> OpenCV

    try {
        // Convert ROS image messages to OpenCV images
        cv_rgb = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
        cv_depth = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::BGR8);
        cv_mask = cv_bridge::toCvCopy(mask_msg, sensor_msgs::image_encodings::BGR8);
        

        // Save the depth image (this will be the reference size)
        depth_img_ = cv_depth->image;

        // Get target size from depth image
        cv::Size target_size = depth_img_.size();

        // Resize RGB and mask images to match the depth image size
        cv::Mat resized_rgb, resized_mask;
        cv::resize(cv_rgb->image, resized_rgb, target_size);
        cv::resize(cv_mask->image, resized_mask, target_size);

        // Save resized images
        current_img_ = resized_rgb;
        mask_img_ = resized_mask;
        // Convert 3-channel mask image to grayscale properly
        cv::cvtColor(resized_mask, mask_img_, cv::COLOR_BGR2GRAY);
        mask_img_.convertTo(mask_img_, CV_8UC1);

        cv::Mat current_copy = current_img_.clone(); // Copia per confronto futuro

        // Se abbiamo un'immagine precedente, eseguiamo feature detection
        if (!prev_img_.empty()) {
            feature_extractor_.featureDetection(prev_img_, current_img_, image_pub_, depth_img_,mask_img_);
        }

        // Salvo la copia dell'immagine corrente come precedente per il prossimo frame
        prev_img_ = current_copy.clone();  
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what()); // Gestione eccezione conversione
        return;
    }
}

/*
   Funzione di pubblicazione immagini elaborate su ROS.
   Parametri:
   - pub: publisher ROS
   - image: immagine OpenCV da pubblicare
   - encoding: encoding ROS dell'immagine (es. "bgr8")
   - frame_id: nome del frame per ROS TF
*/
void OpticalFlowPose::PublishRenderedImage(image_transport::Publisher pub, cv::Mat image, std::string encoding, std::string frame_id) {
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = frame_id;

    // Creazione del messaggio immagine ROS
    const sensor_msgs::ImagePtr rendered_image_msg = cv_bridge::CvImage(header, encoding, image).toImageMsg();
    pub.publish(rendered_image_msg); // Pubblicazione
}

/*
   Funzione per visualizzare punti dinamici e statici sulla corrente immagine e pubblicare.
   Parametri:
   - good_old: punti feature nel frame precedente
   - good_new: punti feature nel frame corrente
   - dynamic: vettore di booleani che indica se il punto è dinamico o statico
   - current: immagine corrente su cui disegnare
*/
void OpticalFlowPose::recoverPose(const std::vector<cv::Point2f> &good_old,
                                  const std::vector<cv::Point2f> &good_new,
                                  const std::vector<bool>& dynamic,
                                  cv::Mat &current)
{
    // Loop sui punti feature per disegnare cerchi sull'immagine
    for (size_t i = 0; i < good_new.size(); ++i)
    {
        if (dynamic[i])
            cv::circle(current, good_new[i], 3, cv::Scalar(0, 255, 0), -1);  // Verde per punti dinamici
        else 
            cv::circle(current, good_new[i], 3, cv::Scalar(255, 0, 0), -1);  // Blu per punti statici
    }

    // Controllo formato immagine: deve essere CV_8UC3
    if (!(current.channels() == 3 && current.depth() == CV_8U)) {
        cv::cvtColor(current, current, cv::COLOR_BGR2RGB); // Conversione se necessario
    }

    // Pubblicazione dell'immagine con i punti disegnati
    PublishRenderedImage(image_pub_, current, "bgr8", "endoscope");
}
