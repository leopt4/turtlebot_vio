#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension, Float64MultiArray

class VelocityConverter:
    def __init__(self):
        rospy.init_node('velocity_converter')
        self.cmd_vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        self.wheel_vel_pub = rospy.Publisher('/turtlebot/kobuki/commands/wheel_velocities', Float64MultiArray, queue_size=10)
        self.d = 0.235   # Distance between wheels (example value, adjust according to your robot)
        self.r = 0.035   # Radius of wheels (example value, adjust according to your robot)

    def cmd_vel_callback(self, msg):
        v = msg.linear.x
        w = -msg.angular.z
        v_left = (v + (w * self.d) / 2) / self.r
        v_right = (v - (w * self.d) / 2) / self.r

        wheel_vel_msg = Float64MultiArray()
        wheel_vel_msg.layout.dim.append(MultiArrayDimension())
        wheel_vel_msg.layout.dim[0].size = 2
        wheel_vel_msg.layout.dim[0].stride = 1
        wheel_vel_msg.layout.dim[0].label = "wheel_velocities"
        wheel_vel_msg.data = [v_left, v_right]

        self.wheel_vel_pub.publish(wheel_vel_msg)

if __name__ == '__main__':
    try:
        converter = VelocityConverter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
