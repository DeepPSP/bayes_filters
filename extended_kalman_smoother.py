"""

"""

import sys
import numpy as np
import scipy.linalg as linalg
from copy import deepcopy
from typing import Union, Optional, Any, Callable
from numbers import Real

from .extended_kalman_filter import ExtendedKalmanFilter


np.set_printoptions(precision=5,suppress=True)


__all__ = [
    "ExtendedKalmanSmoother"
]


class ExtendedKalmanSmoother(ExtendedKalmanFilter):
    r"""
    Implements an extended Kalman smoother (EKS).

    .. math:: to write

    Parameters:
    -----------
    dim_x: int,
        number of state variables for the Kalman filter. For example, if
        you are tracking the position and velocity of an object in two
        dimensions, dim_x would be 4.
    dim_z: int,
        number of of measurement inputs. For example, if the sensor
        provides you with position in (x,y), dim_z would be 2.
    dim_u: int,
        number of control inputs
    --------------------------------
    state_transition_func: callable,
        the state transition function
    measurement_func: callable,
        the measurement function, usually identity
    jac_state: callable,
        jacobian of the state transition function
    jac_measurement: callable,
        jacobian of the measurement function
    --------------------------------
    verbose: int,
        the extent for printing intermediate results
    
    Attributes:
    -----------
    x : np.array (of shape (dim_x, 1))
        state estimate vector
    P : np.array (matrix, of shape (dim_x, dim_x))
        covariance matrix
    x_prior : np.array (of shape (dim_x, 1))
        prior (predicted) state estimate. The *_prior and *_post attributes
        are for convienence; they store the  prior and posterior of the
        current epoch. Read Only.
    P_prior : np.array (matrix, of shape (dim_x, dim_x))
        prior (predicted) state covariance matrix. Read Only.
    x_post : np.array (of shape (dim_x, 1))
        posterior (updated) state estimate. Read Only.
    P_post : np.array (matrix, of shape (dim_x, dim_x))
        posterior (updated) state covariance matrix. Read Only.
    --------------------------------
    state_transition_mat: np.ndarray (matrix, of shape (dim_x,dim_x)),
        the state transition matrix
    control_transition_mat: np.ndarray (matrix, of shape (dim_x,dim_u)),
        the control transition matrix
    measurement_mat: np.ndarray (matrix, of shape (dim_z, dim_x)),
        the measurement (observation) matrix, usually the identity matrix
    --------------------------------
    innovation: np.ndarray (of shape (dim_z,1)),
        the innovation signal (or pre-fit residual)
    innovation_covar:np.ndarray (matrix, of shape (dim_z, dim_z)),
        innovation (or pre-fit residual) covariance
    kalman_gain: np.ndarray (matrix, of shape (dim_x,dim_z)),
        the Kalman gain matrix
    -------------------------------
    process_noise_covar: np.ndarray (matrix, of shape (dim_x, dim_x)),
        the state noise covariance matrix
    measurement_noise_covar: np.ndarray (matrix, of shape (dim_z, dim_z)),
        the measurement noise covariance matrix
    """
    def __init__(self, dim_x:int, dim_z:int, dim_u:int, **kwargs):
        """

        """
        super().__init__(dim_x, dim_z, dim_u, **kwargs)


    def initialize(self, init_x, init_P, state_transition_mat, measurement_mat, process_noise_covar, measurement_noise_covar, init_u=None, control_transition_mat=None):
        return super().initialize(init_x, init_P, state_transition_mat, measurement_mat, process_noise_covar, measurement_noise_covar, init_u=init_u, control_transition_mat=control_transition_mat)


    def predict(self, u=None, state_transition_mat=None, process_noise_covar=None, control_transition_mat=None):
        return super().predict(u=u, state_transition_mat=state_transition_mat, process_noise_covar=process_noise_covar, control_transition_mat=control_transition_mat)


    def update(self, z, measurement_mat=None, measurement_noise_covar=None):
        return super().update(z, measurement_mat=measurement_mat, measurement_noise_covar=measurement_noise_covar)


    def predict_update(self, z, u=None, state_transition_mat=None, measurement_mat=None, process_noise_covar=None, measurement_noise_covar=None, control_transition_mat=None):
        return super().predict_update(z, u=u, state_transition_mat=state_transition_mat, measurement_mat=measurement_mat, process_noise_covar=process_noise_covar, measurement_noise_covar=measurement_noise_covar, control_transition_mat=control_transition_mat)


    def __repr__(self):
        """

        """
        pass


    def __str__(self):
        """

        """
        pass

    