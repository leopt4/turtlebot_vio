#!/usr/bin/python
import tf
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA 

class Environement:
    def __init__(self, odom_topic) -> None:
        self.current_pose = None
        # PUBLISHERS
        self.broadcaster = tf.TransformBroadcaster()
        # self.tf_world_base_footprint_pub = rospy.Publisher('~point_marker', tf, queue_size=1)
        self.point_marker_pub = rospy.Publisher('~point_marker', Marker, queue_size=1)

        # SUBSCRIBERS
        self.odom_sub       = rospy.Subscriber(odom_topic, Odometry, self.get_odom) 

        # TIMERS
        # Timer for velocity controller
        # rospy.Timer(rospy.Duration(0.001), self.publish_tf)


    # Odometry callback: Gets current robot pose and stores it into self.current_pose
    def get_odom(self, odom):
        _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, 
                                                            odom.pose.pose.orientation.y,
                                                            odom.pose.pose.orientation.z,
                                                            odom.pose.pose.orientation.w])
        self.current_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw])

    def publish_tf(self, event):
        if node.current_pose is not None:
            # Define the translation and rotation for the inverse TF (base_footprint to world)
            translation = (self.current_pose[0], self.current_pose[1], 0) # Set the x, y, z coordinates
            quaternion = tf.transformations.quaternion_from_euler(0, 0, self.current_pose[2])  # Convert euler angles to quaternion
            rotation = (quaternion[0], quaternion[1], quaternion[2], quaternion[3])
            
            # Publish the inverse TF from world to base_footprint
            self.broadcaster.sendTransform(
                translation,
                rotation,
                rospy.Time.now(),
                "turtlebot/kobuki/base_footprint",
                "world_ned"
            )


if __name__ == '__main__':
    # Create an instance of the ground truth class
    rospy.init_node('ground_truth_publisher')
    node = Environement('/turtlebot/kobuki/odom_ground_truth')
    # Sleep briefly to ensure the TF is published
    rospy.spin()


