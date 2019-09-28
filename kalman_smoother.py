"""

"""

import sys
import numpy as np
from scipy import linalg
from scipy import stats as ss
from copy import deepcopy
from typing import Union, Optional, Any, Callable, Tuple
from numbers import Real
from utils.units_converter import ArrayLike

from .base import BaseKalmanFilter


class FixedLagSmoother(BaseKalmanFilter):
    """

    Methods that need not override:
        initialize
        predict
        update
        predict_update

    """
    def __init__(self, dim_x:int, dim_z:int, dim_u:int, lag:int, verbose=0):
        """

        """
        super().__init__(
            dim_x=dim_x,
            dim_z=dim_z,
            dim_u=dim_u,
            verbose=verbose
        )
        self.x_s = np.zeros((dim_x, 1), dtype=float)  # smoothed state
        self.count = 0  # t in wiki
        self.lag = lag
        self.x_smoothed = []


    def smooth(self, z:ArrayLike, u:Optional[ArrayLike]=None) -> None:
        """

        a single smooth step, whose number is stored in self.count

        Parameters:
        -----------
        z: array_like,
            the measurement vector at current step
        u: array_like, optional,
            the control input vector

        Returns:
        --------
        None
        """
        self.predict_update(z,u)
        self.x_smoothed.append(self.x)  # self.x_smoothed now has length self.count+1

        P_multiplier = (self.state_transition_mat - np.dot(self.kalman_gain,self.measurement_mat)).T
        kalman_gain_multiplier = np.dot(self.measurement_mat.T, self.inv_innovation_covar)

        if self.count >= self.lag:
            P_smoothed = self.P_prior.copy()  # P^{(0)}
            # the N-1 new variables, in reversed order compared to wiki
            for i in range(1,self.lag):
                kalman_gain_smoothed = np.dot(P_smoothed, kalman_gain_multiplier)  # K^{(i)}
                self.x_smoothed[self.count-i] = self.x_smoothed[self.count-i] + np.dot(kalman_gain_smoothed, self.innovation)
                P_smoothed = np.dot(P_smoothed, P_multiplier)  # P^{(i)}, used in the next loop

        self.count += 1


    def __repr__(self):
        """

        """
        pass


    def __str__(self):
        """

        """
        pass



class RTSSmoother(BaseKalmanFilter):
    """

    Rauch-Tung-Striebel (RTS) smoother, a fixed interval smoother

    """
    def __init__(self, dim_x:int, dim_z:int, dim_u:int, interval:int, verbose:int=0):
        """

        """
        super().__init__(
            dim_x=dim_x,
            dim_z=dim_z,
            dim_u=dim_u,
            verbose=verbose
        )
        self.interval = interval
        self.x_smoothed = []


    def smooth(self):
        """

        """
        pass


    def __repr__(self):
        """

        """
        pass


    def __str__(self):
        """

        """
        pass



class MBFSmoother(BaseKalmanFilter):
    """

    modified Bryson-Frazier (MBF) smoother

    """
    def __init__(self, dim_x:int, dim_z:int, dim_u:int, interval:int, verbose:int=0):
        """

        """
        super().__init__(
            dim_x=dim_x,
            dim_z=dim_z,
            dim_u=dim_u,
            verbose=verbose
        )
        self.interval = interval


    def smooth(self, z:ArrayLike, u:Optional[ArrayLike]=None) -> None:
        """

        """
        pass


    def __repr__(self):
        """

        """
        pass


    def __str__(self):
        """

        """
        pass



class MVSmoother(BaseKalmanFilter):
    """

    mininum-variance (MV) smoother

    """
    def __init__(self, dim_x:int, dim_z:int, dim_u:int, interval:int, verbose:int=0):
        """

        """
        super().__init__(
            dim_x=dim_x,
            dim_z=dim_z,
            dim_u=dim_u,
            verbose=verbose
        )
        self.interval = interval


    def smooth(self, z:ArrayLike, u:Optional[ArrayLike]=None) -> None:
        """

        """
        pass


    def __repr__(self):
        """

        """
        pass


    def __str__(self):
        """

        """
        pass
