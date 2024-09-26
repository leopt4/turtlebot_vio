#ifndef IMAGE_HANDLER_H
#define IMAGE_HANDLER_H

#include "ros/ros.h"
#include <ros/package.h>
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

    // // Following field is stored solely for debug purpose (Kindly remove while actual implementataion)
    // cv::Mat img;
};

class imgBuffer
{
private:
    int bufferLength_;
    std::deque<imageData> buffer_;
    int last_classID;

public:
    imgBuffer(int bufferLength) : bufferLength_(bufferLength) {}

    void push(imageData &data)
    {
        if (data.frameID == 0)
        {
            int classID = 0;
            for (int i = 0; i < data.keypoints.size(); i++)
            {
                data.keypoints[i].class_id = classID++;
            }
            last_classID = classID;
        };

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

    imageData getLastCapturedFrameData()
    {
        return (buffer_.back());
    };

    int getNumberofMembers()
    {
        return (buffer_.size());
    };

    void replace(imageData &data)
    {
        for (auto it = buffer_.begin(); it != buffer_.end(); ++it)
        {
            if (it->frameID == data.frameID)
            {
                *it = data; // Replace the element at the current iterator position
                return;     // Exit the function after replacing the first occurrence
            }
        }
        std::cout << "No element with frameID " << data.frameID << " found in the buffer." << std::endl;
    }

    int getLastClassID()
    {
        return (last_classID);
    }

    void updateLastClassID(int &value)
    {
        last_classID = value;
    }

    void incrementLastClassID()
    {
        last_classID++;
    }
};

class imageHandler
{
public:
    ros::NodeHandle nh_;
    ros::Subscriber odom_sub_, img_in_;
    ros::Publisher feature2D_pub_;

    ros::Timer timer_;

    double thresholdTime_;
    double thresholdOdometry_;

    bool ODOM_TRIGGER = true;
    bool TIME_TRIGGER = false;

    imgBuffer buffer_;

    std::string directory_; // Directory for saving the images (while debugging)

    Ptr<cv::Feature2D> detector_; // Feature detector
    Ptr<cv::BFMatcher> matcher_;

    int current_frame_ID = -1;

    imageHandler(ros::NodeHandle &nh, double thresholdTime, double thresholdOdometry)
        : nh_(nh), thresholdTime_(thresholdTime), thresholdOdometry_(thresholdOdometry), buffer_(3)
    {
        odom_sub_ = nh.subscribe("/odom", 10, &imageHandler::imuCallback, this);
        img_in_ = nh.subscribe("/turtlebot/kobuki/realsense/color/image_color", 10, &imageHandler::imgCallback, this);

        // Initialize feature detector and matcher
        detector_ = cv::ORB::create(250, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
        matcher_ = cv::BFMatcher::create();

        // Timer for getting a new frame
        timer_ = nh.createTimer(ros::Duration(thresholdTime_), &imageHandler::timerCallback, this);

        // Publisher for the Feature2D Positions
        feature2D_pub_ = nh.advertise<turtlebot_vio::Keypoint2DArray>("/feature2D", 10);

        std::string packagePath = ros::package::getPath("turtlebot_vio");
        std::string pkg_directory = "/opencv_tests/";
        directory_ = packagePath + pkg_directory;
        std::cout << directory_ << std::endl;
    };

    // Image callback method
    void imgCallback(const sensor_msgs::ImageConstPtr &msg);

    // IMU callback method
    void imuCallback(const nav_msgs::Odometry::ConstPtr &msg);

    // Timer callback method
    void timerCallback(const ros::TimerEvent &);

private:
    void imagePreprocessing(cv::Mat &img);
    void detectAndMatchFeatures(cv::Mat &img);
    void ransac();
    void updateKeypointClassID();
    void publish2DFeatureMap();
    // void visualizeKeypointClassIDs(const imageData &frameData);
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

        // std::string originalImagePath = directory_ + "/original_" + std::to_string(current_frame_ID) + ".png";
        // cv::imwrite(originalImagePath, img);

        imagePreprocessing(img);

        // std::string processedImagePath = directory_ + "/processed_" + std::to_string(current_frame_ID) + ".png";
        // cv::imwrite(processedImagePath, img);

        ROS_INFO("-------Frame (ID- %d) : Captured and Processed Successfully-------", current_frame_ID);

        ros::Time start_time = ros::Time::now();

        detectAndMatchFeatures(img);
        ransac();
        updateKeypointClassID();
        // visualizeKeypointClassIDs(buffer_.getLastCapturedFrameData());
        publish2DFeatureMap();

        ros::Time end_time = ros::Time::now();
        ros::Duration duration = end_time - start_time;

        ROS_INFO("-------Frame (ID- %d) : 2D Features Published (Execution Time : %f)-------", current_frame_ID, duration.toSec());

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

void imageHandler::detectAndMatchFeatures(cv::Mat &img)
{
    // for Lowe's Ratio test
    const float ratio_thresh = 0.7f;
    imageData newFrameInfo;
    newFrameInfo.frameID = current_frame_ID;
    // newFrameInfo.img = img.clone();

    detector_->detectAndCompute(img, cv::noArray(), newFrameInfo.keypoints, newFrameInfo.descriptors);

    // Check if there is only one frame in the buffer
    if (current_frame_ID == 0)
    {
        buffer_.push(newFrameInfo);
        return; // Return empty map since there's no matching to do
    }
    else
    {
        // Iterate over each frame in the buffer
        for (int i = 0; i < buffer_.getNumberofMembers(); ++i)
        {
            imageData frameData = buffer_.get(i);
            // Match descriptors between the current frame and the frame in the buffer
            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher_->knnMatch(newFrameInfo.descriptors, frameData.descriptors, knn_matches, 2);

            if (knn_matches.empty())
            {
                // Handling the case where no matches are found
                continue; // Skip the current iteration
            }
            else
            {
                // Apply Lowe's ratio test
                std::vector<cv::DMatch> good_matches;
                for (size_t j = 0; j < knn_matches.size(); j++)
                {
                    if (knn_matches[j][0].distance < ratio_thresh * knn_matches[j][1].distance)
                    {
                        good_matches.push_back(knn_matches[j][0]);
                    }
                }
                // Store the good matches and their corresponding frame ID
                if (!good_matches.empty())
                {
                    newFrameInfo.matchingMap.insert(std::make_pair(frameData.frameID, good_matches));
                }
                else
                {
                    continue;
                }
            }
            matcher_->clear();
        }
    }

    // Following block is solely for debug purpose (Kindly remove while actual implementataion)
    /*
    for (int i = 0; i < buffer_.getNumberofMembers(); ++i)
    {
        imageData frameData = buffer_.get(i);

        // Assuming img is the current frame image and frameData.descriptors is the descriptor of the frame in the buffer
        // std::vector<cv::DMatch> matches = matchedFeatures[frameData.frameID];
        std::vector<cv::DMatch> matches = newFrameInfo.matchingMap[frameData.frameID];

        // Draw matches
        cv::Mat img_matches;
        cv::drawMatches(img, newFrameInfo.keypoints, frameData.img, frameData.keypoints, matches, img_matches);

        // Save the visualized matches
        std::string visualizationPath = directory_ + "/matches_" + std::to_string(current_frame_ID) + "_" + std::to_string(frameData.frameID) + ".png";
        cv::imwrite(visualizationPath, img_matches);
    }
    */

    // Store the current frame information
    buffer_.push(newFrameInfo);
    return;
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
            if (currentFramePoints.size() > 0 && bufferFramePoints.size() > 0)
            {
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
                buffer_.replace(currentFrameData);

                // Following block is solely for debug purpose (Kindly remove while actual implementataion)

                // std::vector<cv::DMatch> matches = currentFrameData.matchingMap[frameData.frameID];

                // // Draw matches
                // cv::Mat img_matches;
                // cv::drawMatches(currentFrameData.img, currentFrameData.keypoints, frameData.img, frameData.keypoints, matches, img_matches);

                // // Save the visualized matches
                // std::string visualizationPath = directory_ + "/matches_after_ransac" + std::to_string(current_frame_ID) + "_" + std::to_string(frameData.frameID) + ".png";
                // cv::imwrite(visualizationPath, img_matches);
            }
            else
            {
                continue;
            }
        };
    }
    else
    {
        return;
    }
}

void imageHandler::updateKeypointClassID()
{
    if (current_frame_ID == 0)
    {
        return;
    }
    else
    {
        // Get the current frame
        imageData currentFrameData = buffer_.getLastCapturedFrameData();

        // Initialize a classID to keep track of the last assigned class ID for each keypoint
        int lastAssignedClassID = buffer_.getLastClassID();

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
                    int newClassID = lastAssignedClassID++;
                    currentFrameData.keypoints[queryIdx].class_id = newClassID;
                    frameData.keypoints[trainIdx].class_id = newClassID; // Update the previous frame's keypoint as well
                }
            }
            buffer_.replace(frameData);
        }

        // Replace the current frame data in the buffer with the updated data
        buffer_.replace(currentFrameData);
        buffer_.updateLastClassID(lastAssignedClassID);
        return;
    }
}

void imageHandler::publish2DFeatureMap()
{
    // Get the current frame data
    imageData currentFrameData = buffer_.getLastCapturedFrameData();

    // Prepare the message
    turtlebot_vio::Keypoint2DArray msg;

    // Filter keypoints with class ID not equal to -1
    std::vector<cv::KeyPoint> filteredKeypoints;
    std::copy_if(currentFrameData.keypoints.begin(), currentFrameData.keypoints.end(), std::back_inserter(filteredKeypoints),
                 [](const cv::KeyPoint &kp)
                 { return kp.class_id != -1; });

    // Resize the message keypoints based on the filtered keypoints
    msg.keypoints.resize(filteredKeypoints.size());

    // Copy filtered keypoints to the message
    for (size_t i = 0; i < filteredKeypoints.size(); ++i)
    {
        msg.keypoints[i].class_id = filteredKeypoints[i].class_id;
        msg.keypoints[i].x = filteredKeypoints[i].pt.x;
        msg.keypoints[i].y = filteredKeypoints[i].pt.y;
    }

    // Publish the message
    feature2D_pub_.publish(msg);
}

/*
void imageHandler::visualizeKeypointClassIDs(const imageData &frameData)
{
    cv::Mat imgWithKeypoints;
    cv::cvtColor(frameData.img, imgWithKeypoints, cv::COLOR_GRAY2BGR); // Convert to BGR for drawing

    for (const auto &keypoint : frameData.keypoints)
    {
        int classID = keypoint.class_id;
        if (classID != -1)
        {
            cv::circle(imgWithKeypoints, keypoint.pt, 3, cv::Scalar(0, 255, 0), -1);                                                      // Draw a circle around the keypoint
            cv::putText(imgWithKeypoints, std::to_string(classID), keypoint.pt, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1); // Label the keypoint with its class ID
        }
    }

    // Save or display the image with keypoints and class IDs
    std::string visualizationPath = directory_ + "/keypoints_classID_" + std::to_string(frameData.frameID) + ".png";
    cv::imwrite(visualizationPath, imgWithKeypoints);
}
*/

#endif // IMAGE_HANDLER_H