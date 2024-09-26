import numpy as np
import rospy
from .AngleHandler import *

class Magnetometer:
    def __init__(self, Rk):
        """
        Constructor of the OdomData class.

        :param:
        """
        self.heading        = None          # [rad]
        self.stamp          = None          # [rostime]
        self.newData        = False         # Flag presenting got new data
        self.Rk             = Rk            # covariance of heading noise

    def read_magnetometer(self, mag):
        """
        Read encoder method includes updating heading reading

        :param odom: mag mess 
        :return True: if get displacement
        :return False: not enough encoder reading to compute displacement
        """
        self.heading    = mag
        self.stamp      = rospy.Time.now() 
        self.newData    = True
        return True

    def get_magnetometer(self):
        """
        Get Heading

        :return displacement, Rk: mean heading and its covariance matrix.
        """
        return np.array([normalize_angle(self.heading)]).reshape(1,1), self.Rk
    
