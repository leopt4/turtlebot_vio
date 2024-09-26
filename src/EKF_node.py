#!/usr/bin/python
import tf
import rospy
import numpy as np
from config import *
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose as PoseMsg
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from untils.EKF_3DOF_InputDisplacement_Heading import *

from untils.Odometry import *
from untils.Magnetometer import *

from turtlebot_graph_slam.srv import ResetFilter, ResetFilterResponse

from turtlebot_graph_slam.msg import keyframe

class EKF:
    def __init__(self) -> None:
        # Init using sensors
        Qk          = np.diag(np.array([STD_ODOM_X_VELOCITY ** 2, STD_ODOM_Y_VELOCITY ** 2, np.deg2rad(STD_ODOM_ROTATE_VELOCITY) ** 2])) 
        Rk          = np.diag([np.deg2rad(STD_MAG_HEADING)**2])

        self.current_pose           = None
        self.xk           = np.zeros((3, 1))        # Robot pose in the k frame
        self.Pk           = np.zeros((3, 3))        # Robot covariance in the k frame  
        self.yawOffset    = 0.0
        self.ekf_filter   = None
        self.x_map        = np.zeros((3, 1))        # Robot pose in the map frame
        self.x_frame_k    = np.zeros((3, 1))        # k frame pose in the map frame
        
        self.x_map_op     = np.zeros((3, 1))        # Optimized Robot pose in the map frame
        self.x_frame_k_op = np.zeros((3, 1))        # optimized k frame pose in the map frame

        self.initial_current_pose = None

        self.mode         = MODE
        
        self.odom   = OdomData(Qk)
        self.mag    = Magnetometer(Rk)
        self.poseArray = PoseArray()
        self.poseArrayOptimized = PoseArray()

        # PUBLISHERS   
        self.key_frame_pub   = rospy.Publisher(PUB_KEYFRAME_DEADRECKONING_TOPIC, PoseArray, queue_size=10)
        # Publisher for sending Odometry
        self.odom_pub           = rospy.Publisher(PUB_ODOM_TOPIC, Odometry, queue_size=1)
        
        # SUBSCRIBERS
        self.odom_sub               = rospy.Subscriber(SUB_ODOM_TOPIC, JointState, self.get_odom) 

        if self.mode == "SIL":
            self.ground_truth_sub   = rospy.Subscriber(SUB_GROUND_TRUTH_TOPIC, Odometry, self.get_ground_truth)
            self.ground_truth_pub = rospy.Publisher(PUB_GROUND_TRUTH_TOPIC, Odometry, queue_size=10)
        elif self.mode == "HIL":
            self.IMU_sub            = rospy.Subscriber(SUB_IMU_TOPIC, Imu, self.get_IMU) 


        self.optimized_pose_sub = rospy.Subscriber(SUB_OPTIMIZED_TOPIC, keyframe, self.update_optimized_pose)
        self.optimized_odom_pub = rospy.Publisher(PUB_OPTIMIZED_TOPIC, Odometry, queue_size=1)
        
        self.dead_reckoning_pub = rospy.Publisher(PUB_DEAD_RECKONING_TOPIC, Odometry, queue_size=1)        

        

        # if self.mode == "SIL":
        # Move
        while True:
            if self.current_pose is not None:
                self.yawOffset    = self.current_pose[2]
                break
        
        # SERVICES
        self.reset_srv = rospy.Service(SERVICE_RESET_FILTER, ResetFilter, self.reset_filter)
    

        # TIMERS
        # Timer for displacement reset
        # rospy.Timer(rospy.Duration(odom_window), self.reset_filter)

        # Init EKF Filter
        self.ekf_filter = EKF_3DOF_InputDisplacement_Heading(self.xk, self.Pk, self.odom, self.mag)
    
    # Ground Truth callback: Gets current robot pose and stores it into self.current_pose. Besides, get heading as a measurement to update filter
    def get_ground_truth(self, odom):
        _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, 
                                                            odom.pose.pose.orientation.y,
                                                            odom.pose.pose.orientation.z,
                                                            odom.pose.pose.orientation.w])
        
        self.current_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw])
        
        if self.initial_current_pose is None:
            self.initial_current_pose = Pose3D(self.current_pose.copy().reshape(3, 1))
        
        # Get heading as a measurement to update filter
        if self.mag.read_magnetometer(yaw-self.yawOffset) and self.ekf_filter is not None:
            self.ekf_filter.gotNewHeadingData()
    
    # IMU callback: Gets current robot orientation and stores it into self.current_pose. Besides, get heading as a measurement to update filter
    def get_IMU(self, Imu):
        _, _, yaw = tf.transformations.euler_from_quaternion([Imu.orientation.x, 
                                                            Imu.orientation.y,
                                                            Imu.orientation.z,
                                                            Imu.orientation.w])
        
        yaw = -yaw      # Imu in the turtlebot is NEU while I'm using NED

        if (self.mode == "SIL"):
            self.current_pose = np.array([self.current_pose[0], self.current_pose[1], yaw])
        else:
            self.current_pose = np.array([0.0, 0.0, yaw])
            

        # Get heading as a measurement to update filter
        if self.mag.read_magnetometer(yaw-self.yawOffset) and self.ekf_filter is not None:
            self.ekf_filter.gotNewHeadingData()

    # Odometry callback: Gets encoder reading to compute displacement of the robot as input of the EKF Filter.
    # Run EKF Filter with frequency of odometry reading
    def get_odom(self, odom):

        timestamp        = odom.header.stamp

        # Read encoder
        if (self.mode == "HIL" and len(odom.name) == 2) or self.mode == "SIL":
            if self.odom.read_encoder(odom) and self.ekf_filter is not None:
                self.ekf_filter.gotNewEncoderData()

        if self.ekf_filter is not None:
            # Run EKF Filter
            self.xk, self.Pk = self.ekf_filter.Localize(self.xk, self.Pk)
            x_map            = Pose3D.oplus(self.x_frame_k, self.xk)    
            self.x_map       = np.array([x_map[0], x_map[1], x_map[2]]).reshape(3, 1)
            
            # Compounding the optimized pose
            if len(self.poseArrayOptimized.poses) > 0:
                last_x_frame_pose = self.poseArrayOptimized.poses[-1]
                
                
                _,_,last_frame_yaw = tf.transformations.euler_from_quaternion([last_x_frame_pose.orientation.x, 
                                                                last_x_frame_pose.orientation.y, 
                                                                last_x_frame_pose.orientation.z, 
                                                                last_x_frame_pose.orientation.w])
                
                # print(type(last_x_frame_pose.position.x), type(last_x_frame_pose.position.y), type(last_frame_yaw))
                # print(last_x_frame_pose.position.x, last_x_frame_pose.position.y, last_frame_yaw)
  
                self.x_frame_k_op =  np.array([float(last_x_frame_pose.position.x), 
                                               float(last_x_frame_pose.position.y),
                                               float(last_frame_yaw)]).reshape(3, 1)
            else:
                self.x_frame_k_op = self.x_frame_k.copy()
            
            x_map_op    = Pose3D.oplus(self.x_frame_k_op, self.xk)
            self.x_map_op    = np.array([x_map_op[0], x_map_op[1], x_map_op[2]]).reshape(3, 1)

            # Publish rviz
            self.odom_path_pub(timestamp)
            self.optimized_odom_path_pub(timestamp)
            self.dead_reckoning_path_pub(timestamp)
            
            if self.mode == "SIL" and self.initial_current_pose is not None:
                self.ground_tuth_odom_pub(timestamp)
            
            self.publish_tf_map(timestamp)
            
            self.poseArray.header.stamp = timestamp
            self.poseArray.header.frame_id = FRAME_MAP
            self.key_frame_pub.publish(self.poseArray)

            if self.mode == "HIL":
                self.publish_tf_cam(timestamp)

    # Reset state and covariance of th EKF filter
    def reset_filter(self, request):
        # print("x position: ", np.round(self.xk[0], 2))
        # print("y position: ", np.round(self.xk[1], 2))
        # print("Heading: ", np.round(self.xk[2], 2) * 180 / math.pi)
        # self.xk           = self.current_pose.reshape(3,1)
        self.yawOffset    += self.xk[2]
        self.xk           = np.zeros((3, 1))
        self.Pk           = np.zeros((3, 3))

        self.x_frame_k    = self.x_map
        
        pose = PoseMsg()
        pose.position.x = self.x_frame_k[0].copy()
        pose.position.y = self.x_frame_k[1].copy()
        pose.position.z = 0.0

        quaternion = tf.transformations.quaternion_from_euler(0, 0, float((self.x_frame_k[2, 0])))

        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]

        self.poseArray.poses.append(pose)
        self.poseArrayOptimized.poses.append(pose)

        return ResetFilterResponse(request.reset_filter_requested)
    
    
    def update_optimized_pose(self, keyframe):
        num_of_poses_present_already = len(self.poseArrayOptimized.poses)
        if num_of_poses_present_already >= len(keyframe.keyframePoses.poses):
            print("case 1")
            for i in range(len(keyframe.keyframePoses.poses)):
                self.poseArrayOptimized.poses[i] = keyframe.keyframePoses.poses[i]
        else:
            print("case 2")
            for i in range(len(self.poseArrayOptimized.poses)):
                self.poseArrayOptimized.poses[i] = keyframe.keyframePoses.poses[i]
            
            
    # Publish Filter results
    def odom_path_pub(self, timestamp):
        # Transform theta from euler to quaternion
        quaternion = tf.transformations.quaternion_from_euler(0, 0, float((self.xk[2, 0])))  # Convert euler angles to quaternion

        # Publish predicted odom
        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = FRAME_MAP
        odom.child_frame_id = FRAME_PREDICTED_BASE


        odom.pose.pose.position.x = self.xk[0]
        odom.pose.pose.position.y = self.xk[1]

        odom.pose.pose.orientation.x = quaternion[0]
        odom.pose.pose.orientation.y = quaternion[1]
        odom.pose.pose.orientation.z = quaternion[2]
        odom.pose.pose.orientation.w = quaternion[3]

        odom.pose.covariance = list(np.array([[self.Pk[0, 0], self.Pk[0, 1], 0, 0, 0, self.Pk[0, 2]],
                                [self.Pk[1, 0], self.Pk[1,1], 0, 0, 0, self.Pk[1, 2]],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [self.Pk[2, 0], self.Pk[2, 1], 0, 0, 0, self.Pk[2, 2]]]).flatten())

        # odom.twist.twist.linear.x = self.v
        # odom.twist.twist.angular.z = self.w

        self.odom_pub.publish(odom)

        tf.TransformBroadcaster().sendTransform((float(self.xk[0, 0]), float(self.xk[1, 0]), 0.0), quaternion, timestamp, odom.child_frame_id, odom.header.frame_id)
            
    # Publish Filter results
    def dead_reckoning_path_pub(self, timestamp):
        # Transform theta from euler to quaternion
        quaternion = tf.transformations.quaternion_from_euler(0, 0, float((self.x_map[2, 0])))  # Convert euler angles to quaternion

        # Publish predicted odom
        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = FRAME_MAP
        odom.child_frame_id = FRAME_DEAD_RECKONING_BASE


        odom.pose.pose.position.x = self.x_map[0]
        odom.pose.pose.position.y = self.x_map[1]

        odom.pose.pose.orientation.x = quaternion[0]
        odom.pose.pose.orientation.y = quaternion[1]
        odom.pose.pose.orientation.z = quaternion[2]
        odom.pose.pose.orientation.w = quaternion[3]


        self.dead_reckoning_pub.publish(odom)    
    
    def optimized_odom_path_pub(self, timestamp):
        # Transform theta from euler to quaternion
        quaternion = tf.transformations.quaternion_from_euler(0, 0, float((self.x_map_op[2, 0])))  # Convert euler angles to quaternion

        # Publish predicted odom
        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = FRAME_MAP
        odom.child_frame_id = FRAME_OPTIMIZE


        odom.pose.pose.position.x = self.x_map_op[0]
        odom.pose.pose.position.y = self.x_map_op[1]

        odom.pose.pose.orientation.x = quaternion[0]
        odom.pose.pose.orientation.y = quaternion[1]
        odom.pose.pose.orientation.z = quaternion[2]
        odom.pose.pose.orientation.w = quaternion[3]

        # odom.pose.covariance = list(np.array([[self.Pk[0, 0], self.Pk[0, 1], 0, 0, 0, self.Pk[0, 2]],
        #                         [self.Pk[1, 0], self.Pk[1,1], 0, 0, 0, self.Pk[1, 2]],
        #                         [0, 0, 0, 0, 0, 0],
        #                         [0, 0, 0, 0, 0, 0],
        #                         [0, 0, 0, 0, 0, 0],
        #                         [self.Pk[2, 0], self.Pk[2, 1], 0, 0, 0, self.Pk[2, 2]]]).flatten())

        # odom.twist.twist.linear.x = self.v
        # odom.twist.twist.angular.z = self.w

        self.optimized_odom_pub.publish(odom)

        # tf.TransformBroadcaster().sendTransform((float(self.x_map_op[0, 0]), float(self.x_map_op[1, 0]), 0.0), quaternion, timestamp, odom.child_frame_id, odom.header.frame_id)

    def ground_tuth_odom_pub(self, timestamp):
        
        initial_tf = Pose3D.ominus(self.initial_current_pose)
        ground_truth_pose_map = Pose3D.oplus(initial_tf, self.current_pose.copy().reshape(3, 1))
        
        # Transform theta from euler to quaternion
        quaternion = tf.transformations.quaternion_from_euler(0, 0, float((ground_truth_pose_map[2, 0])))  # Convert euler angles to quaternion

        # Publish predicted odom
        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = FRAME_MAP
        odom.child_frame_id = FRAME_PREDICTED_BASE


        odom.pose.pose.position.x = ground_truth_pose_map[0]
        odom.pose.pose.position.y = ground_truth_pose_map[1]

        odom.pose.pose.orientation.x = quaternion[0]
        odom.pose.pose.orientation.y = quaternion[1]
        odom.pose.pose.orientation.z = quaternion[2]
        odom.pose.pose.orientation.w = quaternion[3]
        
        self.ground_truth_pub.publish(odom)
    
    
    def publish_tf_map(self, timestamp):
        x_map = self.x_map.copy()

        # Define the translation and rotation for the inverse TF (base_footprint to world)
        translation = (x_map[0], x_map[1], 0) # Set the x, y, z coordinates

        quaternion = tf.transformations.quaternion_from_euler(0, 0, x_map[2])  # Convert euler angles to quaternion
        rotation = (quaternion[0], quaternion[1], quaternion[2], quaternion[3])
        
        # Publish the inverse TF from world to base_footprint
        tf.TransformBroadcaster().sendTransform(
            translation,
            rotation,
            timestamp,
            FRAME_BASE,
            FRAME_MAP
        )

    def publish_tf_cam(self, timestamp):
        # Define the translation and rotation for the inverse TF (base_footprint to world)
        translation = (0, 0, 0) # Set the x, y, z coordinates

        quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)  # Convert euler angles to quaternion
        rotation = (quaternion[0], quaternion[1], quaternion[2], quaternion[3])
        
        # Publish the inverse TF from world to base_footprint
        tf.TransformBroadcaster().sendTransform(
            translation,
            rotation,
            timestamp,
            "realsense_link",
            "camera_link"         
        )

        # Transform theta from euler to quaternion
        quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)  # Convert euler angles to quaternion
        translation = (0.0, 0.0, 0.0)
        frame_id = FRAME_DEPTH_CAMERA
        child_frame_id = "sensor"

        tf.TransformBroadcaster().sendTransform(translation, quaternion, timestamp, child_frame_id, frame_id)

    def spin(self):
        pass

if __name__ == '__main__':
    rospy.init_node('EKF_node')
    node = EKF()	
    
    rospy.spin()