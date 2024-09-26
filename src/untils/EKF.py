from .GaussianFilter import *
import numpy as np
from .AngleHandler import *
class EKF(GaussianFilter):
    """
    Extended Kalman Filter class. Implements the :class:`GaussianFilter` interface for the particular case of the Extended Kalman Filter.
    """
    def __init__(self, x0, P0, *args):
        """
        Constructor of the EKF class.

        :param x0: initial mean state vector
        :param P0: initial covariance matrix
        :param args: arguments to be passed to the parent class
        """
        super().__init__(x0, P0, *args)  # call parent constructor
        
        self.encoderData = False
        self.headingData = False

    def f(self, xk_1, uk): # motion model
        """
        Motion model of the EKF **to be overwritten by the child class**.

        :param xk_1: previous mean state vector
        :param uk: input vector
        :return xk_bar, Pk_bar: predicted mean state vector and its covariance matrix
        """
        pass

    def Jfx(self, xk_1, uk):
        """
        Jacobian of the motion model with respect to the state vector. **Method to be overwritten by the child class**.

        :param xk_1: Linearization point. By default the linearization point is the previous state vector taken from a class attribute.
        :return: Jacobian matrix
        """
        pass

    def Jfw(self, xk_1, uk):
        """
        Jacobian of the motion model with respect to the noise vector. **Method to be overwritten by the child class**.

        :param xk_1: Linearization point. By default the linearization point is the previous state vector taken from a class attribute.
        :return: Jacobian matrix
        """
        pass

    def h(self, xk):  # observation model
        """
        The observation model of the EKF is given by:

        .. math::
            z_k=h(x_k,v_k)
            :label: eq-EKF-observation-model

        This method computes the mean of this direct observation model. Therefore it does not depend on v_k since it is
        a zero mean Gaussian noise.

        :param xk: mean of the predicted state vector. By default it is taken from the class attribute.
        :return: expected observation vector
        """
        pass

    def Prediction(self, uk, Qk, xk_1=None, Pk_1=None):
        """
        Prediction step of the EKF. It calls the motion model and its Jacobians to predict the state vector and its covariance matrix.

        :param uk: input vector
        :param Qk: covariance matrix of the noise vector
        :param xk_1: previous mean state vector. By default it is taken from the class attribute. Otherwise it updates the class attribute.
        :param Pk_1: covariance matrix of the previous state vector. By default it is taken from the class attribute. Otherwise it updates the class attribute.
        :return xk_bar, Pk_bar: predicted mean state vector and its covariance matrix. Also updated in the class attributes.
        """

        # KF equations begin here
        if self.encoderData == True:
            # Predict states
            xk_bar = self.f(xk_1, uk)
            # Predict covariance
            Ak          = self.Jfx(xk_1, uk)
            Wk          = self.Jfw(xk_1)
        
            Pk_bar = Ak @ Pk_1 @ Ak.T + Wk @ Qk @ Wk.T

            self.encoderData = False
        else:
            xk_bar = xk_1
            Pk_bar = Pk_1

        return xk_bar, Pk_bar

    def Update(self, zk, Rk, xk_bar, Pk_bar, Hk, Vk):
        """
        Update step of the EKF. It calls the observation model and its Jacobians to update the state vector and its covariance matrix.

        :param zk: observation vector
        :param Rk: covariance matrix of the noise vector
        :param xk_bar: predicted mean state vector.
        :param Pk_bar: covariance matrix of the predicted state vector.
        :param Hk: Jacobian of the observation model with respect to the state vector.
        :param Vk: Jacobian of the observation model with respect to the noise vector.
        :return xk,Pk: updated mean state vector and its covariance matrix. Also updated in the class attributes.
        """

        if self.headingData == True:
            # Compute Kalman gain
            Kk          = Pk_bar @ Hk.T @ np.linalg.inv(Hk @ Pk_bar @ Hk.T + Vk @ Rk @ Vk.T)

            # Compute updated state and covariance
            xk          = xk_bar + Kk @ normalize_angle(zk - self.h(xk_bar))
            I           = np.diag(np.ones(len(xk_bar)))
            Pk          = (I - Kk @ Hk) @ Pk_bar @ (I - Kk @ Hk).T

            self.headingData = False
        else:
            xk          = xk_bar
            Pk          = Pk_bar
   
        return xk, Pk

    def gotNewEncoderData(self):
        self.encoderData = True

    def gotNewHeadingData(self):
        self.headingData = True