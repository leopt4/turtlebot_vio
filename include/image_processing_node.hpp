#ifndef IMAGE_HANDLER_H
#define IMAGE_HANDLER_H

#include "ros/ros.h"
#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <iostream>
#include <vector>
#include <deque>

namespace imageHandler
{

    struct imageData
    {
        int frameID; // ID of the frame in the
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        std::map<int, std::vector<cv::DMatch>> matchingMap;
        cv::Mat tfFromLastFrame;
    };

    class imgBuffer
    {
    private:
        int bufferLength_;
        std::deque<imageData> buffer_;

    public:
        imgBuffer(int bufferLength) : bufferLength_(bufferLength) {}

        void push(const imageData &data)
        {
            if (buffer_.size() >= bufferLength_)
            {
                buffer_.pop_front();
            };
            buffer_.push_back(data);
        };

        imageData get(int index)
        {
            if (index < 0 || index >= buffer_.size())
            {
                throw std::out_of_range("Index out of range");
            };
            return buffer_[index];
        };

        void setBufferLength(int newLength)
        {
            bufferLength_ = newLength;
            while (buffer_.size() > bufferLength_)
            {
                buffer_.pop_front();
            };
        };

        int getBufferLength() const
        {
            return bufferLength_;
        };

        int getLastCapturedFrameID() const
        {
            return (buffer_.back().frameID);
        };

        int getNumberofMembers() const
        {
            return (buffer_.size());
        };
    };

};

class imageHandler
{
public:
    ros::NodeHandle &nh_;
    ros::Subscriber imu_preint_, img_in_;
    ros::Timer timer_;

    double thresholdTime_;
    double thresholdOdometry_;

    bool ODOM_TRIGGER = false;
    bool TIME_TRIGGER = false;

    cv::Ptr<cv::FeatureDetector> detector_;  // Feature detector
    cv::Ptr<cv::DescriptorMatcher> matcher_; // Descriptor matcher

    imageHandler(ros::NodeHandle &nh, double thresholdTime, double thresholdOdometry)
        : nh_(nh), thresholdTime_(thresholdTime), thresholdOdometry_(thresholdOdometry)
    {
        imu_preint_ = nh.subscribe("/turtlebot/kobuki/odom_ground_truth", 1, &imageHandler::imuCallback, this);
        img_in_ = nh.subscribe("image topic", 1, &imageHandler::imgCallback, this);

        // Initialize feature detector and matcher
        detector_ = cv::xfeatures2d::SIFT::create();
        matcher_ = cv::BFMatcher::create(cv::NORM_L2);

        // Timer for getting a new frame
        timer_ = nh.createTimer(ros::Duration(thresholdTime_), &imageHandler::timerCallback, this);

        imgBuffer storageHandler(15);
    };

    // Image callback method
    void imgCallback(const sensor_msgs::ImageConstPtr &msg);

    // IMU callback method
    void imuCallback(const nav_msgs::Odometry::ConstPtr &msg);

    // Timer callback method
    void timerCallback(const ros::TimerEvent &);

private:
    void imagePreprocessing(cv::Mat &img);
    std::map<int, std::vector<cv::DMatch>> detectAndMatchFeatures(cv::Mat &img);
    void triangulateFeatures(std::map<int, std::vector<cv::DMatch>> &matchedFeatures);
    void ransac();
    void publish3DPositions();

    std::vector<cv::KeyPoint> all_good_features_; // Store all good features
    std::vector<cv::Point3f> world_points_;       // Store 3D world points
    std::vector<cv::Point2f> image_points_;       // Store 2D image points
};

void imageHandler::imgCallback(const sensor_msgs::ImageConstPtr &msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    };
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    };

    cv::Mat img = cv_ptr->image;
    imagePreprocessing(img);

    if (ODOM_TRIGGER && TIME_TRIGGER)
    {
        std::map<int, std::vector<cv::DMatch>> matchingMap detectAndMatchFeatures(img);
        triangulateFeatures(matchingMap);
        ransac();
        publish3DPositions();

        // Reset triggers for next frame
        ODOM_TRIGGER = false;
        TIME_TRIGGER = false;
    };
};

void imageHandler::imuCallback(const nav_msgs::Odometry::ConstPtr &msg) {

};

void imageHandler::timerCallback(const ros::TimerEvent &)
{
    TIME_TRIGGER = true;
};

void imageHandler::imagePreprocessing(cv::Mat &img)
{
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(img, img, cv::Size(5, 5), 0);
};

std::map<int, std::vector<cv::DMatch>> imageHandler::detectAndMatchFeatures(cv::Mat &img)
{
    // Detect features in the current frame
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector_->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

    // Initialize a map to store matched features and their IDs
    std::map<int, std::vector<cv::DMatch>> matchedFeatures;

    // Check if there is only one frame in the buffer
    if (storageHandler.getNumberofMembers() == 0)
    {
        // Handle the case where there is only one frame available
        // This might involve special processing or simply storing the current frame
        // For now, let's assume we just store the current frame without matching
        imageData newFrameInfo = {
            0,
            keypoints,
            descriptors,
            matchedFeatures,
        };
        storageHandler.push(newFrameInfo);
        return matchedFeatures; // Return empty map since there's no matching to do
    }

    // Iterate over each frame in the buffer
    for (int i = 0; i < storageHandler.getNumberofMembers(); ++i)
    {
        imageData frameData = storageHandler.get(i);

        // Match descriptors between the current frame and the frame in the buffer
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher_->knnMatch(descriptors, frameData.descriptors, knn_matches, 2);

        // Apply Lowe's ratio test
        std::vector<cv::DMatch> good_matches;
        for (size_t j = 0; j < knn_matches.size(); j++)
        {
            if (knn_matches[j][0].distance < 0.7 * knn_matches[j][1].distance)
            {
                good_matches.push_back(knn_matches[j][0]);
            }
        }

        // Store the good matches and their corresponding frame ID
        if (!good_matches.empty())
        {
            matchedFeatures[frameData.frameID] = good_matches;
        }
    }

    // Calculate the transformation since the last frame captured
    cv::Mat tfFromLastFrame = calculateTransformation(img, storageHandler.get(storageHandler.getNumberofMembers() - 1).keypoints);

    // Store the current frame information including the transformation
    imageData newFrameInfo = {storageHandler.getLastCapturedFrameID() + 1, keypoints, descriptors, matchedFeatures, tfFromLastFrame};
    storageHandler.push(newFrameInfo);

    return matchedFeatures;
}

void imageHandler::triangulateFeatures() {
    // Implement triangulation logic here
    // This involves using the matched features and their 3D positions
};

void imageHandler::ransac() {
    // Implement RANSAC for outlier detection
    // This involves using the matched features and their 3D positions
};

void imageHandler::publish3DPositions() {
    // Publish 3D positions of features on a topic
};

#endif // IMAGE_HANDLER_H