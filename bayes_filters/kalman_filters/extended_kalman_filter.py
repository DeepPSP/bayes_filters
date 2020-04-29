"""
Module: kalman_filters
File: extended_kalman_filter.py

Remarks:
"""

import sys
import numpy as np
import scipy.linalg as linalg
from copy import deepcopy
from typing import Union, Optional, Any, Callable
from numbers import Real

from .base import BaseKalmanFilter, ArrayLike


np.set_printoptions(precision=5,suppress=True)


__all__ = [
    "ExtendedKalmanFilter",
]


class ExtendedKalmanFilter(BaseKalmanFilter):
    r"""
    Implements an extended Kalman filter (EKF).

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

    def __init__(self, dim_x:int, dim_z:int, dim_u:int, verbose:int=0, **kwargs):
        """

        """
        super().__init__(dim_x, dim_z, dim_u, verbose, **kwargs)
        self.state_transition_func = None
        self.measurement_func = None
        self.jac_state = None
        self.jac_state_control_input = None
        self.jac_state_noise = None
        self.jac_measurement = None
        self.jac_measurement_noise = None
        self.state_params = None
        self.measurement_params = None


    def initialize(self, init_x:ArrayLike,
                   init_P:ArrayLike,
                   state_transition_func:Callable[[np.ndarray,np.ndarray,dict],np.ndarray],
                   measurement_func:Callable[[np.ndarray,dict],np.ndarray],
                   jac_state:Callable[[np.ndarray,np.ndarray,dict],np.ndarray],
                   jac_measurement:Callable[[np.ndarray,dict],np.ndarray],
                   state_params:dict,
                   measurement_params:dict,
                   process_noise_covar:ArrayLike,
                   measurement_noise_covar:ArrayLike,
                   jac_state_control_input:Optional[Callable[[np.ndarray,np.ndarray,dict],np.ndarray]]=None,
                   jac_state_noise:Optional[Callable[[np.ndarray,np.ndarray,dict],np.ndarray]]=None,
                   jac_measurement_noise:Optional[Callable[[np.ndarray,dict],np.ndarray]]=None,
                   init_u:Optional[ArrayLike]=None):
        """ initialize the whole system

        Paramters:
        ----------
        init_x: array_like,
        init_P: array_like,
        state_transition_func: callable,
        measurement_func: callable
        jac_state: callable,
        jac_measurement: callable,
        state_params: dict,
        measurement_params: dict,
        process_noise_covar: array_like,
        measurement_noise_covar: array_like,
        jac_state_control_input: callable, optional,
        jac_state_noise: callable, optional,
        jac_measurement_noise: callable, optional,
        init_u: array_like, optional,
        """
        super().initialize(
            init_x=init_x,
            init_P=init_P,
            state_transition_mat=np.eye(self.dim_x, dtype=float),
            measurement_mat=np.zeros((self.dim_z,self.dim_x), dtype=float),
            process_noise_covar=process_noise_covar,
            measurement_noise_covar=measurement_noise_covar,
            init_u=init_u,
            control_transition_mat=None
        )
        self.state_transition_func = state_transition_func
        self.measurement_func = measurement_func
        self.jac_state = jac_state
        self.jac_state_control_input = jac_state_control_input
        self.jac_state_noise = jac_state_noise
        self.jac_measurement = jac_measurement
        self.jac_measurement_noise = jac_measurement_noise
        self.state_params = state_params
        self.measurement_params = measurement_params

        return self


    def predict(self, u:Optional[ArrayLike]=None,
                state_transition_func:Optional[Callable[[np.ndarray,np.ndarray,dict],np.ndarray]]=None,
                jac_state:Optional[Callable[[np.ndarray,np.ndarray,dict],np.ndarray]]=None,
                jac_state_control_input:Optional[Callable[[np.ndarray,np.ndarray,dict],np.ndarray]]=None,
                jac_state_noise:Optional[Callable[[np.ndarray,np.ndarray,dict],np.ndarray]]=None,
                state_params:Optional[dict]=None,
                process_noise_covar:Optional[ArrayLike]=None):
        """ make prediction

        Paramters:
        ----------
        u: array_like, optional,
        state_transition_func: callable, optional,
        jac_state: callable, optional,
        jac_state_control_input: callable, optional,
        jac_state_noise: callable, optional,
        state_params: dict, optional,
        process_noise_covar: array_like, optional,
        """
        try:
            if u is not None:
                self.u = np.array(u, dtype=float).reshape((self.dim_u,1))
            if process_noise_covar is not None:
                self.process_noise_covar = np.array(process_noise_covar, dtype=float).reshape((self.dim_x,self.dim_x))
        except:
            raise ValueError('please check the sizes of the input vectors and matrics')
        
        if state_transition_func is not None:
            self.state_transition_func = state_transition_func
        if jac_state is not None:
            self.jac_state = jac_state
        if jac_state_control_input is not None:
            self.jac_state_control_input = jac_state_control_input
        if jac_state_noise is not None:
            self.jac_state_noise = jac_state_noise
        if state_params is not None:
            self.state_params = state_params

        # make prediction
        self.x_prior = self.state_transition_func(self.x, self.u, self.state_params)
        _state_transition_mat = self.jac_state(self.x, self.u, self.state_params)
        # if self.jac_state_control_input is not None:
        #     _control_transition_mat = self.jac_state_control_input(self.x, self.u, self.state_params)
        self.P_prior = np.dot(_state_transition_mat, np.dot(self.P, _state_transition_mat.T)) + self.process_noise_covar

        if self.verbose >= 1:
            print('***** before the prediction step,')
            print('x = {0},\n P = {1}'.format(self.x, self.P))
            print('***** after the prediction step,')
            print('x_prior = {0},\n P_prior = {1}'.format(self.x_prior, self.P_prior))

        return self


    def update(self, z:ArrayLike,
               measurement_func:Optional[Callable[[np.ndarray,dict],np.ndarray]]=None,
               jac_measurement:Optional[Callable[[np.ndarray,dict],np.ndarray]]=None,
               jac_measurement_noise:Optional[Callable[[np.ndarray,dict],np.ndarray]]=None,
               measurement_params:Optional[dict]=None,
               measurement_noise_covar:Optional[ArrayLike]=None):
        """ update the whole system

        Paramters:
        ----------
        measurement_func: callable, optional,
        jac_measurement: callable, optional,
        jac_measurement_noise: callable, optional,
        measurement_params: dict, optional,
        measurement_noise_covar: array_like, optional,
        """
        try:
            self.z = np.array(z, dtype=float).reshape((self.dim_z,1))
            if measurement_noise_covar is not None:
                self.measurement_noise_covar = np.array(measurement_noise_covar, dtype=float).reshape((self.dim_z,self.dim_z))
        except:
            raise ValueError('please check the sizes of the input vectors and matrices')
        
        if measurement_func is not None:
            self.measurement_func = measurement_func
        if jac_measurement is not None:
            self.jac_measurement = jac_measurement
        if jac_measurement_noise is not None:
            self.jac_measurement_noise = jac_measurement_noise
        if measurement_params is not None:
            self.measurement_params = measurement_params

        # update
        self.innovation = self.z - self.measurement_func(self.x_prior, self.measurement_params)
        _measurement_mat = self.jac_measurement(self.x_prior, self.measurement_params)
        self.innovation_covar = np.dot(_measurement_mat, np.dot(self.P_prior, _measurement_mat.T)) + self.measurement_noise_covar
        self.inv_innovation_covar = np.linalg.inv(self.innovation_covar)
        self.kalman_gain = np.dot(self.P_prior, np.dot(_measurement_mat.T, self.inv_innovation_covar))
        self.x = self.x_prior + np.dot(self.kalman_gain, self.innovation)
        self.x_post = deepcopy(self.x)
        self.P = np.dot(self._identity_mat-np.dot(self.kalman_gain,_measurement_mat), self.P_prior)
        self.P_post = deepcopy(self.P)

        if self.verbose >= 1:
            print('***** for the update step,')
            print('the observation z =', z)
            print('innovation =', self.innovation)
            print('innovation_covar =', self.innovation_covar)
            print('kalman_gain =', self.kalman_gain)
            print('x_post =', self.x_post)
            print('P_post =', self.P_post)
        
        return self


    def predict_update(self, z:ArrayLike,
                       u:Optional[ArrayLike]=None,
                       state_transition_func:Optional[Callable[[np.ndarray,np.ndarray,dict],np.ndarray]]=None,
                       measurement_func:Optional[Callable[[np.ndarray,dict],np.ndarray]]=None,
                       jac_state:Optional[Callable[[np.ndarray,np.ndarray,dict],np.ndarray]]=None,
                       jac_state_control_input:Optional[Callable[[np.ndarray,np.ndarray,dict],np.ndarray]]=None,
                       jac_state_noise:Optional[Callable[[np.ndarray,np.ndarray,dict],np.ndarray]]=None,
                       jac_measurement:Optional[Callable[[np.ndarray,dict],np.ndarray]]=None,
                       jac_measurement_noise:Optional[Callable[[np.ndarray,dict],np.ndarray]]=None,
                       state_params:Optional[dict]=None,
                       measurement_params:Optional[dict]=None,
                       process_noise_covar:Optional[ArrayLike]=None,
                       measurement_noise_covar:Optional[ArrayLike]=None):
        """
        Performs the predict/update innovation of the extended Kalman filter.
        
        Parameters:
        -----------
        ref. self.predict and self.update
        """
        return self.predict(
            u=u,
            state_transition_func=state_transition_func,
            jac_state=jac_state,
            jac_state_control_input=jac_state_control_input,
            jac_state_noise=jac_state_noise,
            state_params=state_params,
            process_noise_covar=process_noise_covar
        ).update(
            z=z,
            measurement_func=measurement_func,
            jac_measurement=jac_measurement,
            jac_measurement_noise=jac_measurement_noise,
            measurement_params=measurement_params,
            measurement_noise_covar=measurement_noise_covar
        )
