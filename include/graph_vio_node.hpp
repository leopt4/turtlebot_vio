#ifndef GRAPH_VIO_HANDLER_H
#define GRAPH_VIO_HANDLER_H

#include "ros/ros.h"
#include "turtlebot_vio/Keypoint2DArray.h"
#include "sensor_msgs/Imu.h"
#include "nav_msgs/Odometry.h"
#include <tf/transform_datatypes.h>
#include <geometry_msgs/Quaternion.h>

#include <Eigen/Dense>

// GTSAM related includes.
#include <gtsam/base/Value.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/slam/PriorFactor.h>

#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/CombinedImuFactor.h>

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>

#include <vector>
#include <cstring>
#include <fstream>
#include <iostream>

using namespace std;
using namespace gtsam;

using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X; // Pose3 (r,p,y,x,y,z)

typedef SmartProjectionPoseFactor<Cal3_S2> sppf;

class graph_vio_handler
{
private:
    NonlinearFactorGraph *graph_;

    ISAM2 *isam2_;

    nav_msgs::Odometry initialOdom_;

    Values initial_estimates_;
    Vector6 initial_sigmas_;

    Values results_;

    noiseModel::Robust::shared_ptr velocity_noise_model_;
    noiseModel::Robust::shared_ptr bias_noise_model_;
    imuBias::ConstantBias prior_imu_bias_; // assuming zero initial bias

    NavState prev_state_;
    NavState prop_state_;
    imuBias::ConstantBias prev_bias_;

    Cal3_S2::shared_ptr K_;
    noiseModel::Isotropic::shared_ptr measurementNoise_;
    std::optional<Pose3> body_P_sensor_; // TODO: Calibration (Camera pose in body)

    std::shared_ptr<PreintegratedCombinedMeasurements::Params> p_;
    std::shared_ptr<PreintegratedCombinedMeasurements> preintegrated_;

    bool stopIMUPreint_ = false;

    // Smart Factor Map <landmark ID , Instance of Smart Factor>
    std::map<size_t, sppf::shared_ptr> smartFactors_;
    SmartProjectionParams sppfparams_; // TODO: Tuning (double triangulation threshold)

protected:
    ros::NodeHandle nh_;
    ros::Subscriber feature2D_sub_, imu_sub_, odom_sub_;

public:
    double gravity_;
    bool isam2InUse_;

    int index_ = 0;
    bool added_initialValues_ = false;
    ros::Time lastIMUStamp_;

    graph_vio_handler(ros::NodeHandle &nh);
    ~graph_vio_handler();

    void readGravity();
    void initGraph();
    void initISAM2();
    void addInitialValuestoGraph(const Pose3 &prior_pose, const Vector3 &prior_vel);
    void getImuParams();
    void readCameraParams();
    void initpreintegrated();
    void imuCB(const sensor_msgs::Imu::ConstPtr &imu_msg);
    void featureCB(const turtlebot_vio::Keypoint2DArray::ConstPtr &feature_msg);
    void odomCB(const nav_msgs::Odometry::ConstPtr &msg);
    void solveAndResetGraph();
};

graph_vio_handler::graph_vio_handler(ros::NodeHandle &nh) : nh_(nh), graph_(nullptr), isam2_(nullptr), preintegrated_(nullptr), p_(nullptr), K_(nullptr), measurementNoise_(nullptr), lastIMUStamp_(ros::Time(0))
{
    feature2D_sub_ = nh.subscribe("/feature2D", 10, &graph_vio_handler::featureCB, this);
    imu_sub_ = nh.subscribe("/turtlebot/kobuki/sensors/imu_data", 100, &graph_vio_handler::imuCB, this);
    odom_sub_ = nh.subscribe("/odom", 1, &graph_vio_handler::odomCB, this);

    SmartProjectionParams sppfParams_(LinearizationMode linMode = HESSIAN, DegeneracyMode degMode = ZERO_ON_DEGENERACY);
    readGravity();
    initGraph();
    initISAM2();
    readCameraParams();
    getImuParams();
    initpreintegrated();
}

graph_vio_handler::~graph_vio_handler()
{
    return;
}

void graph_vio_handler::readGravity()
{
    if (nh_.getParam("/graphVIO/gravity", gravity_))
    {
        ROS_INFO_STREAM("Gravity Read!!");
    }
}

void graph_vio_handler::initGraph()
{
    graph_ = new NonlinearFactorGraph();
}

void graph_vio_handler::initISAM2()
{
    if (nh_.getParam("/graphVIO/Use_ISAM2", isam2InUse_))
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
            ISAM2Params isam2Params;
            isam2Params.relinearizeThreshold = relinearizeThreshold;
            isam2Params.relinearizeSkip = relinearizeSkip;
            isam2Params.factorization = factorization;

            // Create ISAM2 instance
            isam2_ = new ISAM2(isam2Params);

            ROS_INFO("ISAM2 initialized with parameters: relinearizeThreshold: %f, relinearizeSkip: %d, factorization: %s",
                     relinearizeThreshold, relinearizeSkip, factorizationStr.c_str());
        }
        else
        {
            ROS_INFO("ISAM2 Not Initialized!! Using LevenbergMarquardt Optimizer!!");
        }
    }
    else
    {
        ROS_ERROR("Failed to get Use_ISAM2 parameter");
    }
}

void graph_vio_handler::odomCB(const nav_msgs::Odometry::ConstPtr &msg)
{
    initialOdom_ = *msg;

    // // Extract position and orientation from the odometry message
    double x = initialOdom_.pose.pose.position.x;
    double y = initialOdom_.pose.pose.position.y;

    Rot3 quat = Rot3::Quaternion(initialOdom_.pose.pose.orientation.w, initialOdom_.pose.pose.orientation.x, initialOdom_.pose.pose.orientation.y, initialOdom_.pose.pose.orientation.z);

    Pose3 prior_pose(quat, Point3(x, y, 0.0));

    double yaw_cov = initialOdom_.pose.covariance[35];
    double x_cov = initialOdom_.pose.covariance[0];
    double y_cov = initialOdom_.pose.covariance[7];


    Eigen::VectorXd sigmas(6);
    sigmas<<0.01, 0.01, yaw_cov, x_cov, y_cov, 0.001;
    initial_sigmas_ << Vector6(sigmas); // TODO: Tuning

    Vector3 prior_vel(0.0, 0.0, 0.0); // Prior velocity

    initial_estimates_.insert(X(index_), prior_pose);
    initial_estimates_.insert(V(index_), prior_vel);
    initial_estimates_.insert(B(index_), prior_imu_bias_);

    ROS_INFO_STREAM("Initial Values read from Odom!!");

    addInitialValuestoGraph(prior_pose, prior_vel);

    // Unsubscribe from the odometry topic after receiving the first message
    odom_sub_.shutdown();
}

void graph_vio_handler::addInitialValuestoGraph(const Pose3 &prior_pose, const Vector3 &prior_vel)
{

    // Add initial noise models (Read Covariance from odom)
    auto pose_noise_model = noiseModel::Diagonal::Sigmas(initial_sigmas_); // TODO: Tuning

    auto gaussian_bias = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(5.0e-2), Vector3::Constant(5.0e-3)).finished()); // TODO: Tuning
    auto huber_bias = noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(1.345), gaussian_bias);                         // TODO: Tuning

    bias_noise_model_ = huber_bias;

    auto gaussian_vel = noiseModel::Isotropic::Sigma(3, 0.01);                                               // TODO: Tuning
    auto huber_vel = noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(1.345), gaussian_vel); // TODO: Tuning

    velocity_noise_model_ = huber_vel;

    // addPrior to the graph
    graph_->addPrior(X(index_), prior_pose, pose_noise_model);
    graph_->addPrior(V(index_), prior_vel, velocity_noise_model_);
    graph_->addPrior(B(index_), prior_imu_bias_, bias_noise_model_);

    added_initialValues_ = true;
    ROS_INFO_STREAM("Initial Values added to the graph!!");

    prev_state_ = NavState(prior_pose, prior_vel);
    prop_state_ = prev_state_;
    prev_bias_ = prior_imu_bias_;
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
    ROS_INFO_STREAM("HELLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLl");

    // TODO: Reading camera pose in body (body_P_sensor_)
}

void graph_vio_handler::getImuParams()
{
    double accel_noise_sigma, gyro_noise_sigma, accel_bias_rw_sigma, gyro_bias_rw_sigma;

    // Read IMU parameters from the parameter server
    if (!nh_.getParam("/graphVIO/IMU/accel_noise_sigma", accel_noise_sigma) ||
        !nh_.getParam("/graphVIO/IMU/gyro_noise_sigma", gyro_noise_sigma) ||
        !nh_.getParam("/graphVIO/IMU/accel_bias_rw_sigma", accel_bias_rw_sigma) ||
        !nh_.getParam("/graphVIO/IMU/gyro_bias_rw_sigma", gyro_bias_rw_sigma))
    {
        ROS_ERROR("Failed to read IMU parameters from the parameter server");
        return;
    }
    else
    {

        p_ = PreintegratedCombinedMeasurements::Params::MakeSharedD((gravity_));
        Matrix33 measured_acc_cov = I_3x3 * pow(accel_noise_sigma, 2);
        Matrix33 measured_omega_cov = I_3x3 * pow(gyro_noise_sigma, 2);
        Matrix33 integration_error_cov = I_3x3 * 1e-8; // TODO: Tuning
        Matrix33 bias_acc_cov = I_3x3 * pow(accel_bias_rw_sigma, 2);
        Matrix33 bias_omega_cov = I_3x3 * pow(gyro_bias_rw_sigma, 2);
        Matrix66 bias_acc_omega_init = I_6x6 * 1e-5; // TODO: Tuning

        // TODO: Add
        // iRb // Camera to IMU Rotation
        // iTb // Camera to IMU Translation
        // p->body_P_sensor = Pose3(iRb, iTb);

        p_->accelerometerCovariance = measured_acc_cov;
        p_->integrationCovariance = integration_error_cov;
        p_->gyroscopeCovariance = measured_omega_cov;
        p_->biasAccCovariance = bias_acc_cov;
        p_->biasOmegaCovariance = bias_omega_cov;
        p_->biasAccOmegaInt = bias_acc_omega_init;
    }
    return;
}

void graph_vio_handler::initpreintegrated()
{
    if (p_ != nullptr)
    {
        preintegrated_ = std::make_shared<PreintegratedCombinedMeasurements>(p_, prior_imu_bias_);
    }
    else
    {
        ROS_ERROR_STREAM("IMU params not obtained, can not create IMUPreintegration!");
    }
    assert(preintegrated_);
}

void graph_vio_handler::imuCB(const sensor_msgs::Imu::ConstPtr &imu_msg)
{

    if ((!stopIMUPreint_) && (lastIMUStamp_ != ros::Time(0)))
    {
        if (p_ != nullptr && graph_ != nullptr && preintegrated_ != nullptr && added_initialValues_)
        {
            double dt = ((imu_msg->header.stamp) - lastIMUStamp_).toSec();
            preintegrated_->integrateMeasurement(Vector3(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z),
                                                 Vector3(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z),
                                                 dt); // Subtstracted Gravity
        }
        else
        {
            ROS_ERROR_STREAM("Some information missing!, can not do IMU preintegration!!");
        }
        preintegrated_->print("Measurements:- ");
    }

    lastIMUStamp_ = imu_msg->header.stamp;
};

void graph_vio_handler::featureCB(const turtlebot_vio::Keypoint2DArray::ConstPtr &feature_msg)
{
    if ((feature_msg->keypoints).size() > 0)
    {
        stopIMUPreint_ = true;
        index_++;

        auto imu_preint = dynamic_cast<const PreintegratedCombinedMeasurements &>(*preintegrated_);
        CombinedImuFactor imu_factor(X(index_ - 1), V(index_ - 1), X(index_), V(index_), B(index_ - 1), B(index_), imu_preint);

        graph_->add(imu_factor);
        imuBias::ConstantBias zero_bias(Vector3(0, 0, 0), Vector3(0, 0, 0)); // TODO: Tuning
        graph_->add(BetweenFactor<imuBias::ConstantBias>(B(index_ - 1), B(index_), zero_bias, bias_noise_model_));

        prop_state_ = preintegrated_->predict(prev_state_, prev_bias_);

        initial_estimates_.insert(X(index_), prop_state_.pose());
        initial_estimates_.insert(V(index_), prop_state_.v());
        initial_estimates_.insert(B(index_), prev_bias_);

        for (const auto &feature : feature_msg->keypoints)
        {
            int classID = feature.class_id;
            if (smartFactors_.count(classID) == 0)
            {
                // If there's no SmartProjectionPoseFactor for this classID, create one
                smartFactors_[classID] = sppf::shared_ptr(new sppf(measurementNoise_, K_, body_P_sensor_, sppfparams_));
                graph_->add(smartFactors_[classID]);
            }

            smartFactors_[classID]->add(Point2(feature.x, feature.y), X(index_));
        }

        solveAndResetGraph();

        prev_state_ = NavState(results_.at<Pose3>(X(index_)), results_.at<Vector3>(V(index_)));

        prev_bias_ = results_.at<imuBias::ConstantBias>(B(index_));

        stopIMUPreint_ = false; // TODO: Change position of this trigger after graph update
        preintegrated_->resetIntegrationAndSetBias(prev_bias_);
    }
    else
    {
        return;
    }
};

void graph_vio_handler::solveAndResetGraph()
{
    if (isam2InUse_)
    {
        ROS_INFO_STREAM("Optimizing the graph using ISAM2!");
        isam2_->update(*graph_, initial_estimates_);
        results_ = isam2_->calculateEstimate();
    }
    else
    {
        ROS_INFO_STREAM("Optimizing the graph using Levenberg-Marquardt Optimizer!");
        LevenbergMarquardtOptimizer optimizer(*graph_, initial_estimates_);
        results_ = optimizer.optimize();
    }

    graph_->resize(0);
    initial_estimates_.clear();
}

#endif // GRAPH_VIO_HANDLER_H
