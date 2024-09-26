from .GaussianFilter import GaussianFilter
import math
from .AngleHandler import *

class GFLocalization(GaussianFilter):
    """
    Map-less localization using a Gaussian filter.
    """
    def __init__(self, x0, P0, *args):
        """
        Constructor.

        :param x0: initial state
        :param P0: initial covariance
        :param index: Named tuple used to relate the state vector, the simulation and the observation vectors (:class:`IndexStruct.IndexStruct`)
        :param kSteps: simulation time steps
        :param robot: Simulated Robot object
        :param args: arguments to be passed to the parent constructor
        """

        self.encoderData = False
        self.headingData = False
        self.featureData = False

        super().__init__(x0, P0, *args)  # call parent constructor

    def GetInput(self):  # get the input from the robot. To be overidden by the child class
        """
        Get the input from the robot. Relates to the motion model as follows:

        .. math::
            x_k &= f(x_{k-1},u_k,w_k) \\\\
            w_k &= N(0,Q_k)
            :label: eq-f-GFLocalization


        **To be overidden by the child class** .

        :return uk, Qk: input and covariance of the motion model
        """
        pass

    def GetMeasurements(self):  # get the measurements from the robot To be overidden by the child class
        """
        Get the measurements from the robot. Corresponds to the observation model:

        .. math::
            z_k &= h(x_{k},v_k) \\\\
            v_k &= N(0,R_k)
            :label: eq-h


        **To be overidden by the child class** .

        :return: zk, Rk, Hk, Vk: observation vector and covariance of the observation noise. Hk is the Observation matrix and Vk is the noise observation matrix.
        """
        pass

    def Localize(self, xk_1, Pk_1):
        """
        Localization iteration. Reads the input of the motion model, performs the prediction step, reads the measurements, performs the update step and logs the results.
        The method also plots the uncertainty ellipse of the robot pose.

        :param xk_1: previous state vector
        :param Pk_1: previous covariance matrix
        :return xk, Pk: updated state vector and covariance matrix
        """

        # Get input to prediction step
        uk, Qk          = self.GetInput()
        # Prediction step
        xk_bar, Pk_bar  = self.Prediction(uk, Qk, xk_1, Pk_1)

        # Get measurement, Heading of the robot
        zk, Rk, Hk, Vk  = self.GetMeasurements()
        # Update step
        xk, Pk          = self.Update(zk, Rk, xk_bar, Pk_bar, Hk, Vk)

        # Normalise heading angle
        xk[2]           = normalize_angle(xk[2])
        return xk, Pk
        # return xk, Pk, xk_bar, zk, Rk

    def normalize_angle(angle):
        """
        Normalize an angle to the range between -π and π.

        Args:
            angle (float): The angle to be normalized, in radians.

        Returns:
            float: The normalized angle.
        """
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
