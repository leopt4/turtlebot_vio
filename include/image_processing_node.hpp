#ifndef IMAGE_HANDLER_H
#define IMAGE_HANDLER_H

#include "ros/ros.h"
#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <sensor_msgs/Image.h>           // Include for sensor_msgs
#include <nav_msgs/Odometry.h>           // Include for nav_msgs
#include <sensor_msgs/image_encodings.h> // Include for sensor_msgs
#include <cv_bridge/cv_bridge.h>
#include <opencv2/flann.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include <iostream>
#include <vector>
#include <deque>

#include "turtlebot_vio/Keypoint2DArray.h"

using namespace cv;
using namespace cv::xfeatures2d;

struct imageData
{
    int frameID; // ID of the frame
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::map<int, std::vector<cv::DMatch>> matchingMap;

    // Following field is stored solely for debug purpose (Kindly remove while actual implementataion)
    cv::Mat img;
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

    int getBufferLength()
    {
        return bufferLength_;
    };

    int getLastCapturedFrameID()
    {
        return (buffer_.back().frameID);
    };

    int getNumberofMembers()
    {
        return (buffer_.size());
    };
};

class imageHandler
{
public:
    ros::NodeHandle &nh_;
    ros::Subscriber imu_preint_, img_in_;
    ros::Publisher feature2D_pub_;

    ros::Timer timer_;

    double thresholdTime_;
    double thresholdOdometry_;

    bool ODOM_TRIGGER = true;
    bool TIME_TRIGGER = false;

    imgBuffer buffer_;

    std::string directory_; // Directory for saving the images (while debugging)

    Ptr<cv::FeatureDetector> detector_;  // Feature detector
    Ptr<cv::DescriptorMatcher> matcher_; // Descriptor matcher

    int current_frame_ID = -1;

    imageHandler(ros::NodeHandle &nh, double thresholdTime, double thresholdOdometry)
        : nh_(nh), thresholdTime_(thresholdTime), thresholdOdometry_(thresholdOdometry), buffer_(5), directory_("/home/patweatharva/ros_work/turtlebot_class/src/turtlebot_vio/opencv_tests")
    {
        imu_preint_ = nh.subscribe("/odom", 1, &imageHandler::imuCallback, this);
        img_in_ = nh.subscribe("/turtlebot/kobuki/realsense/color/image_color", 1, &imageHandler::imgCallback, this);

        // Initialize feature detector and matcher
        detector_ = cv::ORB::create();
        matcher_ = cv::FlannBasedMatcher::create();

        // Timer for getting a new frame
        timer_ = nh.createTimer(ros::Duration(thresholdTime_), &imageHandler::timerCallback, this);

        // Publisher for the Feature2D Positions
        feature2D_pub_ = nh.advertise<turtlebot_vio::Keypoint2DArray>("/feature2D", 10);
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
    void ransac();
    void updateKeypointClassID();
    void publish2DFeatureMap();
};

void imageHandler::imgCallback(const sensor_msgs::ImageConstPtr &msg)
{
    if (ODOM_TRIGGER && TIME_TRIGGER)
    {
        (++current_frame_ID);
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception &e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        };

        cv::Mat img = cv_ptr->image;

        std::string originalImagePath = directory_ + "/original_" + std::to_string(current_frame_ID) + ".png";
        cv::imwrite(originalImagePath, img);

        imagePreprocessing(img);

        std::string processedImagePath = directory_ + "/processed_" + std::to_string(current_frame_ID) + ".png";
        cv::imwrite(processedImagePath, img);

        ROS_INFO("-------Frame (ID- %d) Captured and Processed Successfully-------", current_frame_ID);

        std::map<int, std::vector<cv::DMatch>> matchingMap = detectAndMatchFeatures(img);
        // ransac();
        // publish2DFeatureMap();

        // Reset triggers for next frame
        // ODOM_TRIGGER = false;
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
    // for Lowe's Ratio test
    const float ratio_thresh = 0.7f;

    // Detect features in the current frame
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector_->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

    // Convert descriptors to CV_32F format if necessary
    if (descriptors.type() != CV_32F)
    {
        descriptors.convertTo(descriptors, CV_32F);
    }

    std::cout << "Features and Descriptors Computed" << std::endl;

    // Initialize a map to store matched features and their IDs
    std::map<int, std::vector<cv::DMatch>> matchedFeatures;

    std::cout << "Map for matched feature created" << std::endl;

    // Check if there is only one frame in the buffer
    if (current_frame_ID == 0)
    {
        // Handle the case where there is only one frame available
        // This might involve special processing or simply storing the current frame
        imageData newFrameInfo = {
            0,
            keypoints,
            descriptors,
            matchedFeatures,
            img};
        buffer_.push(newFrameInfo);
        std::cout << "First frame with empty map added to the buffer" << std::endl;
        return matchedFeatures; // Return empty map since there's no matching to do
    }
    else
    {

        // Iterate over each frame in the buffer
        for (int i = 0; i < buffer_.getNumberofMembers(); ++i)
        {
            imageData frameData = buffer_.get(buffer_.getNumberofMembers()-1-i);

            // Match descriptors between the current frame and the frame in the buffer
            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher_->knnMatch(descriptors, frameData.descriptors, knn_matches, 2);

            if (knn_matches.empty())
            {
                // Handling the case where no matches are found
                std::cout << "No matches found between Frame ID-: " << current_frame_ID << " and " << frameData.frameID << std::endl;
                continue; // Skip the current iteration
            }
            else
            {
                std::cout << "Matches found between frame ID" << current_frame_ID << " and " << frameData.frameID << std::endl;
                // Apply Lowe's ratio test
                std::vector<cv::DMatch> good_matches;
                for (size_t j = 0; j < knn_matches.size(); j++)
                {
                    if (knn_matches[j][0].distance < ratio_thresh * knn_matches[j][1].distance)
                    {
                        good_matches.push_back(knn_matches[j][0]);
                    }
                }

                std::cout << "Lowe's ratio applied" << std::endl;
                std::cout << "Size of good_matches" << good_matches.size() << std::endl;
                // Store the good matches and their corresponding frame ID
                if (!good_matches.empty())
                {
                    std::cout << "Good Matches stored for the Frame ID" << current_frame_ID << " and " << frameData.frameID << std::endl;
                    matchedFeatures.insert(std::make_pair(frameData.frameID,good_matches));
                    // matchedFeatures[frameData.frameID] = good_matches;
                    std::cout << "Good Matches Stored in the map" << std::endl;
                }
                else
                {
                    continue;
                }
            }
        }
    }

    std::cout << "Matching Done for Frame ID - " << current_frame_ID << std::endl;

    // Following block is solely for debug purpose (Kindly remove while actual implementataion)
    for (int i = 0; i < buffer_.getNumberofMembers(); ++i)
    {
        imageData frameData = buffer_.get(i);

        // Assuming img is the current frame image and frameData.descriptors is the descriptor of the frame in the buffer
        std::vector<cv::DMatch> matches = matchedFeatures[frameData.frameID];

        // Draw matches
        cv::Mat img_matches;
        cv::drawMatches(img, keypoints, frameData.img, frameData.keypoints, matches, img_matches);

        // Save the visualized matches
        std::string visualizationPath = directory_ + "/matches_" + std::to_string(current_frame_ID) + "_" + std::to_string(frameData.frameID) + ".png";
        cv::imwrite(visualizationPath, img_matches);
    }

    std::cout << "Matching visualisation done for Frame ID - " << current_frame_ID << std::endl;

    // Store the current frame information
    imageData newFrameInfo = {current_frame_ID, keypoints, descriptors, matchedFeatures, img};
    buffer_.push(newFrameInfo);

    std::cout << "New Frame stored in the Buffer - " << current_frame_ID << std::endl;

    return matchedFeatures;
}

void imageHandler::ransac()
{
    if (current_frame_ID > 0)
    {
        int mem = buffer_.getNumberofMembers();
        // Iterate over each frame in the buffer
        for (int i = 0; i < mem - 1; ++i)
        {
            // Assuming the current frame is the last one in the buffer
            imageData currentFrameData = buffer_.get(mem - 1);

            imageData frameData = buffer_.get(mem - 2 - i);

            // Extract matched points
            std::vector<cv::Point2f> currentFramePoints;
            std::vector<cv::Point2f> bufferFramePoints;

            // Convert matches to points
            for (const auto &match : currentFrameData.matchingMap[frameData.frameID])
            {
                currentFramePoints.push_back(currentFrameData.keypoints[match.queryIdx].pt);
                bufferFramePoints.push_back(frameData.keypoints[match.trainIdx].pt);
            }

            // Apply RANSAC
            std::vector<uchar> inliersMask;
            cv::Mat homography = cv::findHomography(bufferFramePoints, currentFramePoints, cv::RANSAC, 3, inliersMask);

            // Filter outliers
            std::vector<cv::DMatch> inliers;
            for (size_t j = 0; j < inliersMask.size(); ++j)
            {
                if (inliersMask[j])
                {
                    inliers.push_back(currentFrameData.matchingMap[frameData.frameID][j]);
                }
            }

            // Update the matching map with inliers
            currentFrameData.matchingMap[frameData.frameID] = inliers;

            // Replace the current frame data in the buffer with the updated data
            buffer_.push(currentFrameData);
        };
    };
}

void imageHandler::updateKeypointClassID()
{
    if (current_frame_ID == 0)
    {
        // Assign sequential integers to each keypoint in the frame
        int classID = 0;
        imageData currentFrameData = buffer_.get(buffer_.getNumberofMembers() - 1);
        for (auto &keypoint : currentFrameData.keypoints)
        {
            keypoint.class_id = classID++;
        }
    }
    else
    {
        // Get the current frame
        imageData currentFrameData = buffer_.get(buffer_.getNumberofMembers() - 1);

        // Initialize a map to keep track of the last assigned class ID for each keypoint
        std::map<int, int> lastAssignedClassID;

        // Iterate over the frames in the buffer in reverse order (from the most recent to the oldest)
        for (int i = buffer_.getNumberofMembers() - 2; i >= 0; --i)
        {
            imageData frameData = buffer_.get(i);

            // Check matches with the current frame
            for (const auto &match : currentFrameData.matchingMap[frameData.frameID])
            {
                int queryIdx = match.queryIdx;
                int trainIdx = match.trainIdx;

                // If the keypoint in the previous frame has a class ID, assign the same to the current keypoint
                if (frameData.keypoints[trainIdx].class_id != -1)
                {
                    currentFrameData.keypoints[queryIdx].class_id = frameData.keypoints[trainIdx].class_id;
                }
                else
                {
                    // If the keypoint in the previous frame does not have a class ID, assign a new unique ID
                    // Use the last assigned class ID for the current frame as the base and increment it
                    int newClassID = lastAssignedClassID[frameData.frameID] + 1;
                    currentFrameData.keypoints[queryIdx].class_id = newClassID;
                    frameData.keypoints[trainIdx].class_id = newClassID; // Update the previous frame's keypoint as well
                    lastAssignedClassID[frameData.frameID] = newClassID; // Update the last assigned class ID for the previous frame
                }
            }
        }
    }
}

void imageHandler::publish2DFeatureMap()
{
    // Get the current frame data
    imageData currentFrameData = buffer_.get(buffer_.getNumberofMembers() - 1);

    // Prepare the message
    turtlebot_vio::Keypoint2DArray msg;
    msg.keypoints.resize(currentFrameData.keypoints.size());

    for (size_t i = 0; i < currentFrameData.keypoints.size(); ++i)
    {
        msg.keypoints[i].class_id = currentFrameData.keypoints[i].class_id;
        msg.keypoints[i].x = currentFrameData.keypoints[i].pt.x;
        msg.keypoints[i].y = currentFrameData.keypoints[i].pt.y;
    }

    // Publish the message
    feature2D_pub_.publish(msg);
}

#endif // IMAGE_HANDLER_H