from .GFLocalization import *
from .EKF import *
from .Pose import *
from .Odometry import *
from .Magnetometer import *

class EKF_3DOF_InputDisplacement_Heading(GFLocalization, EKF, OdomData):
    """
    This class implements an EKF localization filter for a 3 DOF Diffenteial Drive using an input displacement motion model incorporating
    yaw measurements from the compass sensor.
    It inherits from :class:`GFLocalization.GFLocalization` to implement a localization filter, from the :class:`DR_3DOFDifferentialDrive.DR_3DOFDifferentialDrive` class and, finally, it inherits from
    :class:`EKF.EKF` to use the EKF Gaussian filter implementation for the localization.
    """
    def __init__(self, x0, P0, odom, mag, *args):
        """
        Constructor. Creates the list of  :class:`IndexStruct.IndexStruct` instances which is required for the automated plotting of the results.
        Then it defines the inital stawe vecto mean and covariance matrix and initializes the ancestor classes.

        :param kSteps: number of iterations of the localization loop
        :param robot: simulated robot object
        :param args: arguments to be passed to the base class constructor
        """
        self.odomData   = odom
        self.magData    = mag
        self.wheelBase   = 0.235
        self.wheelRadius = 0.035
        # this is required for plotting
        super().__init__(x0, P0, *args)

    def getOdom(self):
        return self.odomData.odom
    
    def f(self, xk_1, uk):
         
        posBk_1 = Pose3D(xk_1[0:3])
        xk_bar  = posBk_1.oplus(uk)
        return xk_bar

    def Jfx(self, xk_1, uk):
         
        posBk_1 = Pose3D(xk_1[0:3])
        J=posBk_1.J_1oplus(uk)
        return J

    def Jfw(self, xk_1):
         
        posBk_1 = Pose3D(xk_1[0:3])
        J = posBk_1.J_2oplus()
        return J

    def h(self, xk):  #:hm(self, xk):
         
        # Obserse the heading of the robot
        h   = xk[2,0]
        return h  # return the expected observations

    def GetInput(self):
        """

        :return: uk,Qk
        """
         
        # Get output of encoder via ReadEncoder() function
        if self.encoderData:
            uk, Qk = self.odomData.get_displacement()
        else:
            uk = None
            Qk = None

        return uk, Qk

    def GetMeasurements(self):  # override the observation model
        """

        :return: zk, Rk, Hk, Vk
        """
        if self.headingData == True: 
            # Read compass sensor
            zk, Rk  = self.magData.get_magnetometer()
        else:
            zk = np.zeros((0,0))
            Rk = np.zeros((0,0))

        # Raise flag got measurement
        if len(zk) != 0:
            # Compute H matrix
            ns      = len(self.xk)
            Hk      = np.zeros((1,ns))
            Hk[0,2] = 1
            # Compute V matrix
            Vk      = np.diag([1.])
        else:
            # Compute H matrix
            ns      = len(self.xk)
            Hk      = np.zeros((0,ns))
            # Compute V matrix
            Vk      = np.zeros((0,0))
        return zk, Rk, Hk, Vk
