import numpy as np
import rospy
from config import *

class Encoder:
    def __init__(self, tag:str):
        self.tag       = tag        # Name tag pf encoder [left/right]
        self.position   = 0.0       # Position          [met]
        self.velocity   = 0.0       # Angular velocity  [rad/sec]
        self.stamp      = None      # Time stamp        [ros time]

class OdomData:
    def __init__(self, Qk):
        """
        Constructor of the OdomData class.

        :param:
        """
        self.newData    = False     # Flag presenting got new synchronized data
        self.mode       = MODE
        
        # Init left and right encoders
        self.rightEncoder   = Encoder('turtlebot/kobuki/wheel_right_joint')
        self.leftEncoder    = Encoder('turtlebot/kobuki/wheel_left_joint')

        self.synchronized_velocity  = [0.0, 0.0]    # Synchronized angular velocity including the left and right encoders [rad/sec]
        self.synchronized_stamp     = None          # Synchronized time stamp [ros time]
        self.deltaT                 = None          # The length of period between this sync data and the last sync data
        self.displacement           = [0.0, 0.0]    # Displacement between this sync data and the last sync data

        self.Qk         = Qk # covariance of displacement noise

    def update_encoder_reading(self, odom):
        """
        Parse encoder measurements
        :param:
        :return True: 
        """
        if self.mode == "SIL":
            # Check if encoder data is for the left wheel
            if self.leftEncoder.tag in odom.name:
                self.leftEncoder.velocity   = odom.velocity[0]
                self.leftEncoder.stamp      = rospy.Time.now() 
            # Check if encoder data is for the right wheel
            elif self.rightEncoder.tag in odom.name:
                self.rightEncoder.velocity  = odom.velocity[0]
                self.rightEncoder.stamp     = rospy.Time.now() 
        elif self.mode == "HIL":
            # Get encoder data for the left wheel
            self.leftEncoder.velocity   = odom.velocity[0]
            self.leftEncoder.stamp      = rospy.Time.now() 
            # Get encoder data for the right wheel
            self.rightEncoder.velocity  = odom.velocity[1]
            self.rightEncoder.stamp     = rospy.Time.now() 
        return True

    def synchronize_encoder_reading(self):
        """
        Synchronize data between the left and right encoders. Because we can not get encoder measurement of both encoders at the same time
        :param:
        :return True: got synchronized measurement
        :retuen False: not get
        """
        # Synchronize encoder data if readings for both wheels are available
        if self.leftEncoder.stamp is not None and self.rightEncoder.stamp is not None:
            next_synchronized_stamp     = 0.5 * ((self.leftEncoder.stamp.secs + self.rightEncoder.stamp.secs) + (self.leftEncoder.stamp.nsecs + self.rightEncoder.stamp.nsecs)/1e9)  
            # Compute period
            if self.synchronized_stamp is not None:
                self.deltaT                 = next_synchronized_stamp - self.synchronized_stamp

            self.synchronized_stamp     = next_synchronized_stamp
            # Synchronize encoder readings here
            # For demonstration, let's assume the readings are already synchronized
            self.synchronized_velocity  = [self.leftEncoder.velocity, self.rightEncoder.velocity]
            # Publish synchronized data or use it in your control algorithm

            # Reset readings for next iteration
            self.leftEncoder.stamp      = None
            self.rightEncoder.stamp     = None
            
            # Get synchronized encoder data
            if self.deltaT is not None:
                return True
        # Not get synchronized encoder data
        return False
    
    def compute_displacement(self):
        """
        Compute displacement between this sync data and the last sync data

        :param: 
        :return uk: displacement
        """
        # Compute displacements of the left and right wheels
        d_L = self.synchronized_velocity[0] * ROBOT_WHEEL_RADIUS * self.deltaT
        d_R = self.synchronized_velocity[1] * ROBOT_WHEEL_RADIUS * self.deltaT
        # Compute displacement of the center point of robot between k-1 and k
        d       = (d_L + d_R) / 2.
        # Compute rotated angle of robot around the center point between k-1 and k
        delta_theta_k   = (-d_R + d_L) / ROBOT_WHEEL_BASE

        # Compute xk from xk_1 and the travel distance and rotated angle. Got the equations from chapter 1.4.1: Odometry 
        uk              = np.array([[d],
                                    [0],
                                    [delta_theta_k]])
        
        return uk
    
    def read_encoder(self, odom):
        """
        Read encoder method includes updating encoder reading, synchronizing them and computing displacement

        :param odom: odom mess 
        :return True: if get displacement
        :return False: not enough encoder reading to compute displacement
        """
        self.update_encoder_reading(odom)

        if self.synchronize_encoder_reading():
            self.displacement = self.compute_displacement()
            return True
        
        return False

    def get_displacement(self):
        """
        Get displacement

        :return displacement, Qk: mean displacement vector and its covariance matrix.
        """
        return self.displacement, self.Qk

