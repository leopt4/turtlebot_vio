#ifndef GRAPH_VIO_HANDLER_H
#define GRAPH_VIO_HANDLER_H

#include "ros/ros.h"
#include <ros/package.h>

#include "turtlebot_vio/Keypoint2DArray.h"
#include "nav_msgs/Odometry.h"
#include <tf/transform_datatypes.h>
#include <geometry_msgs/Quaternion.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Dense>

// GTSAM related includes.
#include <gtsam/base/Value.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/ISAM2Params.h>

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/slam/PriorFactor.h>

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>

// #include "turtlebot_graph_slam/ResetFilter.h"

#include <geometry_msgs/Point.h>
#include <geometry_msgs/Pose.h>

#include <vector>
#include <iostream>
#include <string>
#include <memory>
#include <exception>
#include <stdexcept>

using namespace std;
using namespace gtsam;

using symbol_shorthand::X; // Pose3 (r,p,y,x,y,z)

typedef SmartProjectionPoseFactor<Cal3_S2> sppf;

class graph_vio_handler
{
private:
    std::unique_ptr<NonlinearFactorGraph> graph_;
    std::unique_ptr<ISAM2> isam2_;

    ISAM2Params isam2params_;
    LevenbergMarquardtParams optimizerParams_;

    nav_msgs::Odometry initialOdom_;
    nav_msgs::Odometry currentOdom_;

    Values initial_estimates_;
    Pose3 current_pose_;
    Eigen::Matrix<double, 6, 6> prior_noise_model_;
    Eigen::Matrix<double, 6, 6> pose_noise_model_;

    Values results_;

    Cal3_S2::shared_ptr K_;
    noiseModel::Isotropic::shared_ptr measurementNoise_;
    std::optional<Pose3> body_P_sensor_; // TODO: Calibration (Camera pose in body)

    // Smart Factor Map <landmark ID , Instance of Smart Factor>
    std::map<size_t, sppf::shared_ptr> smartFactors_;
    SmartProjectionParams sppfparams_; // TODO: Tuning (double triangulation threshold)

protected:
    ros::NodeHandle nh_;
    ros::Subscriber feature2D_sub_, odom_sub_;
    ros::Publisher odom_pub_ = nh_.advertise<nav_msgs::Odometry>("/VIOdometry", 10);
    ros::Publisher point_pub_ = nh_.advertise<geometry_msgs::Point>("/3DPoints", 10);
    ros::Publisher point_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/3DPointsMarkers", 10);
    ros::Timer timer_;


public:
    bool saveGraph_ = false;

    bool isam2InUse_;

    bool graph_initialised_ = false;
    bool optimizer_initialised_ = false;

    bool initialSmartFactorsAdded_ = false;
    bool added_initialValues_ = false;

    int index_ = 0;

    bool stopOdom_ = false;

    // ros::ServiceClient client_;

    graph_vio_handler(ros::NodeHandle &nh);
    ~graph_vio_handler();

    void initGraph();
    void initOptimizer();
    void addInitialValuestoGraph();
    void readCameraParams();
    void featureCB(const turtlebot_vio::Keypoint2DArray::ConstPtr &feature_msg);
    void odomCB(const nav_msgs::Odometry::ConstPtr &msg);
    void solveAndResetGraph();
    // void resetOdom();
    void publishOdom();
    void publishFeatures(const ros::TimerEvent &);
};

graph_vio_handler::graph_vio_handler(ros::NodeHandle &nh) : nh_(nh), graph_(nullptr), isam2_(nullptr), K_(nullptr), measurementNoise_(nullptr) //, client_(nh.serviceClient<turtlebot_graph_slam::ResetFilter>("ResetFilter"))
{
    feature2D_sub_ = nh.subscribe("/feature2D", 10, &graph_vio_handler::featureCB, this);
    odom_sub_ = nh.subscribe("/odom", 1, &graph_vio_handler::odomCB, this);


    SmartProjectionParams sppfParams_(LinearizationMode linMode = HESSIAN, DegeneracyMode degMode = ZERO_ON_DEGENERACY);
    initGraph();
    initOptimizer();
    readCameraParams();


    timer_ = nh_.createTimer(ros::Duration(0.1), &graph_vio_handler::publishFeatures, this);
}

graph_vio_handler::~graph_vio_handler()
{
    return;
}

void graph_vio_handler::initGraph()
{
    if (graph_ == nullptr)
    {
        graph_ = std::make_unique<NonlinearFactorGraph>();
        ROS_INFO_STREAM("Nonlinear Factor Graph Initialized!!");
        graph_initialised_ = true;
    }
}

// void graph_vio_handler::resetOdom()
// {
//     turtlebot_graph_slam::ResetFilter srv;
//     srv.request.reset_filter_requested = true;

//     if (client_.call(srv))
//     {
//         ROS_INFO("Odometry Filter (EKF) is succesfully reset.... : %d", (bool)srv.response.reset_filter_response);
//     }
//     else
//     {
//         ROS_ERROR("Failed to call reset Odometry Filter!!!!!!!");
//     };
// }

void graph_vio_handler::initOptimizer()
{
    if (nh_.getParam("/graphVIO/ISAM2/Use_ISAM2", isam2InUse_))
    {
        if (isam2InUse_)
        {
            // Fetch ISAM2 parameters
            double relinearizeThreshold;  // TODO: Tuning
            int relinearizeSkip;          // TODO: Tuning
            std::string factorizationStr; // TODO: Tuning

            if (!nh_.getParam("/graphVIO/ISAM2/params/relinearizeThreshold", relinearizeThreshold))
            {
                ROS_ERROR("Failed to get ISAM2 relinearizeThreshold parameter");
                return;
            }
            if (!nh_.getParam("/graphVIO/ISAM2/params/relinearizeSkip", relinearizeSkip))
            {
                ROS_ERROR("Failed to get ISAM2 relinearizeSkip parameter");
                return;
            }
            if (!nh_.getParam("/graphVIO/ISAM2/params/factorization", factorizationStr))
            {
                ROS_ERROR("Failed to get ISAM2 factorization parameter");
                return;
            }

            // Convert string to gtsam::ISAM2Params::Factorization
            gtsam::ISAM2Params::Factorization factorization;
            if (factorizationStr == "QR")
            {
                factorization = gtsam::ISAM2Params::Factorization::QR;
            }
            else if (factorizationStr == "CHOLESKY")
            {
                factorization = gtsam::ISAM2Params::Factorization::CHOLESKY;
            }
            else
            {
                ROS_ERROR("Invalid factorization type: %s", factorizationStr.c_str());
                return;
            }

            // Initialize ISAM2 with fetched parameters
            isam2params_.factorization = factorization;
            isam2params_.relinearizeSkip = relinearizeSkip;
            isam2params_.setRelinearizeThreshold(relinearizeThreshold);

            // Create ISAM2 instance
            isam2_ = std::make_unique<ISAM2>(isam2params_);
            ROS_INFO_STREAM("ISAM2 Optimizer initialized with parameters!!!!");
        }
        else
        {
            ROS_INFO("ISAM2 Not Initialized!! Using LevenbergMarquardt Optimizer!!");

            // Read LevenbergMarquardt parameters
            double initialLambda, lambdaFactor, lambdaUpperBound, lambdaLowerBound, minModelFidelity, diagonalDamping;
            bool useDiagonalDamping, useLevenbergMarquardt;

            if (!nh_.getParam("/graphVIO/LevenbergMarquardt/params/initialLambda", initialLambda))
            {
                ROS_ERROR("Failed to get LevenbergMarquardt initialLambda parameter");
                return;
            }
            if (!nh_.getParam("/graphVIO/LevenbergMarquardt/params/lambdaFactor", lambdaFactor))
            {
                ROS_ERROR("Failed to get LevenbergMarquardt lambdaFactor parameter");
                return;
            }
            if (!nh_.getParam("/graphVIO/LevenbergMarquardt/params/lambdaUpperBound", lambdaUpperBound))
            {
                ROS_ERROR("Failed to get LevenbergMarquardt lambdaUpperBound parameter");
                return;
            }
            if (!nh_.getParam("/graphVIO/LevenbergMarquardt/params/lambdaLowerBound", lambdaLowerBound))
            {
                ROS_ERROR("Failed to get LevenbergMarquardt lambdaLowerBound parameter");
                return;
            }
            if (!nh_.getParam("/graphVIO/LevenbergMarquardt/params/minModelFidelity", minModelFidelity))
            {
                ROS_ERROR("Failed to get LevenbergMarquardt minModelFidelity parameter");
                return;
            }
            if (!nh_.getParam("/graphVIO/LevenbergMarquardt/params/useDiagonalDamping", useDiagonalDamping))
            {
                ROS_ERROR("Failed to get LevenbergMarquardt useDiagonalDamping parameter");
                return;
            }
            if (!nh_.getParam("/graphVIO/LevenbergMarquardt/params/diagonalDamping", diagonalDamping))
            {
                ROS_ERROR("Failed to get LevenbergMarquardt diagonalDamping parameter");
                return;
            }
            if (!nh_.getParam("/graphVIO/LevenbergMarquardt/params/useLevenbergMarquardt", useLevenbergMarquardt))
            {
                ROS_ERROR("Failed to get LevenbergMarquardt useLevenbergMarquardt parameter");
                return;
            }

            optimizerParams_.setlambdaInitial(initialLambda);        // TODO: Tuning
            optimizerParams_.setlambdaFactor(lambdaFactor);          // TODO: Tuning
            optimizerParams_.setlambdaUpperBound(lambdaUpperBound);  // TODO: Tuning
            optimizerParams_.setlambdaLowerBound(lambdaLowerBound);  // TODO: Tuning
            optimizerParams_.setDiagonalDamping(useDiagonalDamping); // TODO: Tuning

            ROS_INFO_STREAM("LevenbergMarquardt Optimizer initialized with parameters!!");
        }
        optimizer_initialised_ = true;
    }
    else
    {
        ROS_ERROR("Failed to get Use_ISAM2 parameter");
    }
}

void graph_vio_handler::odomCB(const nav_msgs::Odometry::ConstPtr &msg)
{
    if (!stopOdom_)
    {
        currentOdom_ = *msg;

        // // Extract position and orientation from the odometry message
        double x = currentOdom_.pose.pose.position.x;
        double y = currentOdom_.pose.pose.position.y;

        Rot3 quat = Rot3::Quaternion(currentOdom_.pose.pose.orientation.w, currentOdom_.pose.pose.orientation.x, currentOdom_.pose.pose.orientation.y, currentOdom_.pose.pose.orientation.z);

        current_pose_ = Pose3(quat, Point3(x, y, 0.0));

        pose_noise_model_ << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, currentOdom_.pose.covariance[35], currentOdom_.pose.covariance[30], currentOdom_.pose.covariance[31], 0.0,
            0.0, 0.0, currentOdom_.pose.covariance[5], currentOdom_.pose.covariance[0], currentOdom_.pose.covariance[1], 0.0,
            0.0, 0.0, currentOdom_.pose.covariance[11], currentOdom_.pose.covariance[6], currentOdom_.pose.covariance[7], 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

        if (!added_initialValues_)
        {
            initialOdom_ = currentOdom_;
            prior_noise_model_ = pose_noise_model_;

            // ROS_INFO_STREAM("Initial Values read from Odom!!");
        }
        // prior_noise_model << initialOdom_.pose.covariance[0], initialOdom_.pose.covariance[1], initialOdom_.pose.covariance[2], initialOdom_.pose.covariance[3], initialOdom_.pose.covariance[4], initialOdom_.pose.covariance[5],
        //                     initialOdom_.pose.covariance[6], initialOdom_.pose.covariance[7], initialOdom_.pose.covariance[8], initialOdom_.pose.covariance[9], initialOdom_.pose.covariance[10], initialOdom_.pose.covariance[11],
        //                     initialOdom_.pose.covariance[12], initialOdom_.pose.covariance[13], initialOdom_.pose.covariance[14], initialOdom_.pose.covariance[15],initialOdom_.pose.covariance[16], initialOdom_.pose.covariance[17],
        //                     initialOdom_.pose.covariance[18], initialOdom_.pose.covariance[19], initialOdom_.pose.covariance[20], initialOdom_.pose.covariance[21], initialOdom_.pose.covariance[22], initialOdom_.pose.covariance[23],
        //                     initialOdom_.pose.covariance[24], initialOdom_.pose.covariance[25], initialOdom_.pose.covariance[26], initialOdom_.pose.covariance[27],initialOdom_.pose.covariance[28], initialOdom_.pose.covariance[29],
        //                     initialOdom_.pose.covariance[30], initialOdom_.pose.covariance[31],initialOdom_.pose.covariance[32], initialOdom_.pose.covariance[33], initialOdom_.pose.covariance[34], initialOdom_.pose.covariance[35];
    }
}

void graph_vio_handler::addInitialValuestoGraph()
{
    // addPrior to the graph
    initial_estimates_.insert(X(index_), current_pose_);
    graph_->addPrior(X(index_), current_pose_, prior_noise_model_);

    added_initialValues_ = true;
    ROS_INFO_STREAM("Initial Values added to the graph!!");
}

void graph_vio_handler::readCameraParams()
{
    // Define the parameters to read from the parameter server
    double fx, fy, u0, v0, s, pixelSigma;

    // Read the camera parameters from the parameter server
    if (!nh_.getParam("/graphVIO/camera/fx", fx) ||
        !nh_.getParam("/graphVIO/camera/fy", fy) ||
        !nh_.getParam("/graphVIO/camera/u0", u0) ||
        !nh_.getParam("/graphVIO/camera/v0", v0) ||
        !nh_.getParam("/graphVIO/camera/s", s) ||
        !nh_.getParam("/graphVIO/camera/pixelSigma", pixelSigma))
    {
        ROS_ERROR("Failed to read camera parameters from the parameter server");
        return;
    }

    // Create a Cal3_S2 object with the read parameters
    K_ = std::make_shared<gtsam::Cal3_S2>(fx, fy, s, u0, v0);
    ROS_INFO_STREAM("Camera parameters read successfully: fx=" << fx << ", fy=" << fy << ", cx=" << u0 << ", cy=" << v0 << ", skew=" << s);

    measurementNoise_ = noiseModel::Isotropic::Sigma(2, pixelSigma);
    ROS_INFO_STREAM("Measurement Noise model initialized with " << pixelSigma);

    // TODO: Reading camera pose in body (body_P_sensor_)
}

void graph_vio_handler::featureCB(const turtlebot_vio::Keypoint2DArray::ConstPtr &feature_msg)
{
    if ((feature_msg->keypoints).size() > 0)
    {
        // stopOdom_ = true;
        if (!initialSmartFactorsAdded_ && !added_initialValues_)
        {
            addInitialValuestoGraph();

            for (const auto &feature : feature_msg->keypoints)
            {
                int classID = feature.class_id;

                if (smartFactors_.count(classID) == 0)
                {
                    smartFactors_[classID] = sppf::shared_ptr(new sppf(measurementNoise_, K_, body_P_sensor_, sppfparams_));
                }

                try
                {
                    graph_->add(smartFactors_[classID]);

                    // Attempt to add the feature to the SmartProjectionPoseFactor
                    smartFactors_[classID]->add(Point2(feature.x, feature.y), X(index_));
                }
                catch (const std::exception &e)
                {
                    // ROS_ERROR_STREAM("Error adding feature to SmartProjectionPoseFactor: " << e.what());
                    continue; // Skip the current iteration and move to the next feature
                }
            }
            initialSmartFactorsAdded_ = true;
            ROS_INFO_STREAM("Initial Smart Factors added to the Graph.");
        }
        else
        {
            index_++;
            ROS_INFO_STREAM("Current Index: " << index_);

            auto current_pose_noise = noiseModel::Constrained::Covariance(pose_noise_model_);

            initial_estimates_.insert(X(index_), current_pose_);
            graph_->addPrior(X(index_), current_pose_, current_pose_noise);

            for (const auto &feature : feature_msg->keypoints)
            {
                int classID = feature.class_id;
                if (smartFactors_.count(classID) == 0)
                {
                    // If there's no SmartProjectionPoseFactor for this classID, create one
                    smartFactors_[classID] = sppf::shared_ptr(new sppf(measurementNoise_, K_, body_P_sensor_, sppfparams_));
                    graph_->add(smartFactors_[classID]);
                }
                try
                {
                    // Attempt to add the feature to the SmartProjectionPoseFactor
                    smartFactors_[classID]->add(Point2(feature.x, feature.y), X(index_));
                }
                catch (const std::exception &e)
                {
                    // ROS_ERROR_STREAM("Error adding feature to SmartProjectionPoseFactor: " << e.what());
                    continue; // Skip the current iteration and move to the next feature
                }
            }
            if (index_ > 2)
            {
                solveAndResetGraph();
                ROS_INFO_STREAM("Current Pose:  " << results_.at<Pose3>(X(index_)));
                publishOdom();
            }
        }
    }
    else
    {
        return;
    }
};

void graph_vio_handler::solveAndResetGraph()
{
    try
    {
        if (isam2InUse_)
        {
            ROS_INFO_STREAM("Optimizing the graph using ISAM2!");
            isam2_->update(*graph_, initial_estimates_);
            results_ = isam2_->calculateEstimate();

            graph_->resize(0);
            initial_estimates_.clear();
        }
        else
        {
            ROS_INFO_STREAM("Optimizing the graph using Levenberg-Marquardt Optimizer!");

            // Create LevenbergMarquardtOptimizer instance
            LevenbergMarquardtOptimizer optimizer(*graph_, initial_estimates_, optimizerParams_);
            results_ = optimizer.optimize();
        }
    }
    catch (const std::exception &e)
    {
        ROS_ERROR_STREAM("Error occurred while optimizing the graph: " << e.what());
    }
    catch (...)
    {
        ROS_ERROR_STREAM("Unknown error occurred while optimizing the graph.");
    }

    if (saveGraph_)
    {
        std::string file_name = ros::package::getPath("turtlebot_vio") + "/graph_viz/factorGraphViz.dot";
        ROS_INFO_STREAM("Graph viz file" << file_name);
        graph_->saveGraph(file_name, results_);
    }
};

void graph_vio_handler::publishOdom()
{
    // Check if the results_ map contains the current pose estimate
    if (!results_.empty())
    {
        // Retrieve the current pose estimate
        Pose3 current_pose_estimate = results_.at<Pose3>(X(index_));

        // Convert the Pose3 estimate to an Odometry message
        nav_msgs::Odometry odom;
        odom.header.stamp = ros::Time::now();
        odom.header.frame_id = "map";                            // Assuming "odom" is the frame you want to use
        odom.child_frame_id = "turtlebot/kobuki/base_footprint"; // Assuming "base_link" is the child frame

        // Set the position and orientation from the Pose3 estimate
        odom.pose.pose.position.x = current_pose_estimate.translation().x();
        odom.pose.pose.position.y = current_pose_estimate.translation().y();
        odom.pose.pose.position.z = current_pose_estimate.translation().z();

        Quaternion qt = (current_pose_estimate.rotation()).toQuaternion();

        odom.pose.pose.orientation.y = qt.y();
        odom.pose.pose.orientation.x = qt.x();
        odom.pose.pose.orientation.z = qt.z();
        odom.pose.pose.orientation.w = qt.w();

        // Publish the Odometry message
        odom_pub_.publish(odom);
    }
    else
    {
        ROS_WARN("No pose estimate found in results_ for publishing.");
    }
}

void graph_vio_handler::publishFeatures(const ros::TimerEvent &)
{
    if (!results_.empty())
    {
        visualization_msgs::MarkerArray markerArray;

        int markerId = 0;
        // Iterate over all the smart factor instances stored in the map smartFactors_
        for (const auto& smartFactorPair : smartFactors_)
        {
            // Calculate the 3D point for the smart factor using the pose estimates from the results_ map
            auto pointEstimate = (smartFactorPair.second)->point(results_);
            if (pointEstimate)
            {
                // Create a geometry_msgs::Point message
                geometry_msgs::Point pointMsg;
                pointMsg.x = pointEstimate->x();
                pointMsg.y = pointEstimate->y();
                pointMsg.z = pointEstimate->z();

                // Publish the 3D point
                point_pub_.publish(pointMsg);

                visualization_msgs::Marker marker;
                marker.header.frame_id = "map";
                marker.header.stamp = ros::Time::now();
                marker.ns = "points";
                marker.id = markerId++;
                marker.type = visualization_msgs::Marker::SPHERE;
                marker.action = visualization_msgs::Marker::ADD;
                marker.pose.position.x = pointEstimate->x();
                marker.pose.position.y = pointEstimate->y();
                marker.pose.position.z = pointEstimate->z();
                marker.pose.orientation.x = 0.0;
                marker.pose.orientation.y = 0.0;
                marker.pose.orientation.z = 0.0;
                marker.pose.orientation.w = 1.0;
                marker.scale.x = 0.1; // Adjust the scale as needed
                marker.scale.y = 0.1;
                marker.scale.z = 0.1;
                marker.color.a = 1.0; // Alpha
                marker.color.r = 1.0; // Red
                marker.color.g = 0.0; // Green
                marker.color.b = 0.0; // Blue

                markerArray.markers.push_back(marker);

            }
            else
            {
                continue;
            }
        }
        point_marker_pub_.publish(markerArray);
    }
}

#endif // GRAPH_VIO_HANDLER_H
