"""
Base kalman filter,
and basic functions frequently used
"""

import sys
import numpy as np
from numpy import dot, zeros, eye, diag
from math import log, exp, sqrt
from scipy import linalg
from scipy import stats as ss
from copy import deepcopy

from typing import Union, Optional, Any, Callable, Tuple


__all__ = [
    "BaseKalmanFilter",
    "runge_kutta4",
    "mul",
    "mul_pdf",
    "multivariate_multiply",
    "add",
    "NESS",
    "inv_diagonal",
    "outer_product_sum",
]


class BaseKalmanFilter(object):
    r"""
    Base (linear) kalman filter

    .. math::
        \begin{align*}
        \left{
            \mathbf{x}_k = F_k \mathbf{x}_{k-1} + B_k \mathbf{u}_k + \mathbf{w}_k
            \mathbf{z}_k = H_k \mathbf{x}_k + \mathbf{v}_k
        \right.
        \end{align*}
        where $\mathbf{x}$ is the state vector, $\mathbf{u}$ is the (optional) control input vector,
        $F$ is the state transition matrix, $B$ is the control transition matrix,
        $\mathbf{w}$ is the process (state) noise vector, with covariance matrix $Q$;
        $\mathbf{z}$ is the measurement vector, $H$ is the measurement matrix,
        $\mathbf{v}$ is the measurement noise vector, with covariance matrix $R$.

    Parameters:
    -----------
    dim_x: int,
        number of state variables
    dim_z: int,
        number of measurement variables
    dim_u: int,
        number of control input

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
    --------------------------------
    log_likelihood : float
        log-likelihood of the last measurement. Read only.
    likelihood : float
        likelihood of last measurment. Read only.
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
    mahalanobis : float
        mahalanobis distance of the innovation. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.
        Read only.

    References:
    -----------
        [1]. 
        [2]. 

    Note that there are at least 3 ways to do matrix multiplication using numpy, i.e.
    np.dot, np.matmul, @
    As far as I know, np.dot is the fastest. One can do experiments using the following code
    >>> from timeit import timeit
    >>> hehe1 = np.ones((20,9))
    >>> hehe2 = np.ones((9,3))
    >>> print(timeit(lambda :np.dot(hehe1,hehe2), number=100000))
    >>> print(timeit(lambda :np.matmul(hehe1,hehe2), number=100000))
    >>> print(timeit(lambda :hehe1@hehe2, number=100000))
    """

    def __init__(self, dim_x:int, dim_z:int, dim_u:int, verbose:int=0, **kwargs):
        """

        """
        self.dim_x = dim_x  # dim. of states
        self.dim_z = dim_z  # dim. of measurements
        self.dim_u = dim_u  # dim. of control input

        self.x = zeros((dim_x, 1), dtype=float)  # state
        self.P = eye(dim_x, dtype=float)  # uncertainty covariance
        self.u = zeros((dim_u, 1), dtype=float)  # control input
        self.z = np.full(shape=(dim_z,1), fill_value=np.nan, dtype=float)  # measurement

        self.state_transition_mat = eye(dim_x, dtype=float)  # state transition matrix
        self.control_transition_mat = zeros((dim_x,dim_u), dtype=float)
        self.measurement_mat = zeros((dim_z,dim_x), dtype=float)  # measurement matrix

        self.innovation = zeros((dim_z, 1), dtype=float)  # inovation or measurement prefit residual
        self.innovation_covar = zeros((dim_z, dim_z), dtype=float)  # innovation (or pre-fit residual) covariance
        self.inv_innovation_covar = zeros((dim_z, dim_z), dtype=float)
        self.kalman_gain = zeros((dim_x, dim_z), dtype=float)  # kalman gain

        self.process_noise_covar = eye(dim_x, dtype=float)  # process uncertainty
        self.measurement_noise_covar = eye(dim_z, dtype=float)  # state uncertainty

        # these will always be a copy of x, state_covar after predict() is called
        # self.x_prior = self.x.copy()
        # self.P_prior = self.P.copy()
        self.x_prior = None
        self.P_prior = None

        # these will always be a copy of x, state_covar after update() is called
        # self.x_post = self.x.copy()
        # self.P_post = self.P.copy()
        self.x_post = None
        self.P_post = None
        
        # identity matrix. Do not alter this.
        # self._I = eye(dim_x)
        self._identity_mat = eye(dim_x)

        # Only computed only if requested via property
        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        self.initialized = False
        self.verbose = verbose
    

    @property
    def log_likelihood(self) -> float:
        """
        log-likelihood of the last measurement.

        Returns:
        --------
        log_likelihood: float
        """
        if self._log_likelihood is None:
            self._log_likelihood = ss.multivariate_normal.logpdf(x=self.innovation, cov=self.innovation_covar, allow_singular=True)
        return self._log_likelihood


    @property
    def likelihood(self) -> float:
        """
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.

        Returns:
        --------
        likelihood: float
        """
        if self._likelihood is None:
            self._likelihood = exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood


    @property
    def mahalanobis(self) -> float:
        """"
        Mahalanobis distance of measurement. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.

        Returns:
        --------
        mahalanobis: float
        """
        if self._mahalanobis is None:
            self._mahalanobis = sqrt(float(dot(dot(self.innovation.T, self.inv_innovation_covar), self.innovation)))
        return self._mahalanobis


    def initialize(self, init_x:Union[list,tuple,np.ndarray],
                   init_P:Union[list,tuple,np.ndarray],
                   state_transition_mat:Union[list,tuple,np.ndarray],
                   measurement_mat:Union[list,tuple,np.ndarray],
                   process_noise_covar:Union[list,tuple,np.ndarray],
                   measurement_noise_covar:Union[list,tuple,np.ndarray],
                   init_u:Union[list,tuple,np.ndarray,type(None)]=None,
                   control_transition_mat:Union[list,tuple,np.ndarray,type(None)]=None):
        """

        """
        try:
            self.x = np.array(init_x, dtype=float).reshape((self.dim_x,1))
            self.P = np.array(init_P, dtype=float).reshape((self.dim_x,self.dim_x))
            self.state_transition_mat = np.array(state_transition_mat, dtype=float).reshape((self.dim_x,self.dim_x))
            self.measurement_mat = np.array(measurement_mat, dtype=float).reshape((self.dim_z,self.dim_z))
            self.process_noise_covar = np.array(process_noise_covar, dtype=float).reshape((self.dim_x,self.dim_x))
            self.measurement_noise_covar = np.array(measurement_noise_covar, dtype=float).reshape((self.dim_z,self.dim_z))
            if init_u is not None:
                self.u = np.array(init_u, dtype=float).reshape((self.dim_z,1))
            if control_transition_mat is not None:
                self.control_transition_mat = np.array(control_transition_mat, dtype=float).reshape((dim_x,dim_u))
        except:
            raise ValueError('please check the sizes of the input vectors and matrics')
        self.initialized = True
        
        return self


    def predict(self, u:Union[list,tuple,np.ndarray,type(None)]=None,
                state_transition_mat:Union[list,tuple,np.ndarray,type(None)]=None,
                process_noise_covar:Union[list,tuple,np.ndarray,type(None)]=None,
                control_transition_mat:Union[list,tuple,np.ndarray,type(None)]=None):
        """

        """
        try:
            if u is not None:
                self.u = np.array(u, dtype=float).reshape((self.dim_u,1))
            if state_transition_mat is not None:
                self.state_transition_mat = np.array(state_transition_mat, dtype=float).reshape((self.dim_x,self.dim_x))
            if process_noise_covar is not None:
                self.process_noise_covar = np.array(process_noise_covar, dtype=float).reshape((self.dim_x,self.dim_x))
            if control_transition_mat is not None:
                self.control_transition_mat = np.array(control_transition_mat, dtype=float).reshape((self.dim_x,self.dim_u))
        except:
            raise ValueError('please check the sizes of the input vectors and matrics')

        # make prediction
        self.x_prior = dot(self.state_transition_mat,self.x) + dot(self.control_transition_mat,self.u)
        self.P_prior = dot(self.state_transition_mat, dot(self.P, self.state_transition_mat.T)) + self.process_noise_covar

        if self.verbose >= 1:
            print("after prediction, x_prior = {0}, P_prior = {1}".format(self.x_prior, self.P_prior))

        return self


    def update(self, z:Union[list,tuple,np.ndarray],
               measurement_mat:Union[list,tuple,np.ndarray,type(None)]=None,
               measurement_noise_covar:Union[list,tuple,np.ndarray,type(None)]=None):
        """

        """
        try:
            self.z = np.array(z, dtype=float).reshape((self.dim_z,1))
            if measurement_mat is not None:
                self.measurement_mat = np.array(measurement_mat, dtype=float).reshape((self.dim_z,self.dim_z))
            if measurement_noise_covar is not None:
                self.measurement_noise_covar = np.array(measurement_noise_covar, dtype=float).reshape((self.dim_z,self.dim_z))
        except:
            raise ValueError('please check the sizes of the input vectors and matrics')

        # update
        self.innovation = self.z - dot(self.measurement_mat,self.x_prior)
        self.innovation_covar = dot(self.measurement_mat, dot(self.P_prior,self.measurement_mat.T)) + self.measurement_noise_covar
        self.inv_innovation_covar = np.linalg.inv(self.innovation_covar)
        self.kalman_gain = dot(self.P_prior, dot(self.measurement_mat.T,self.inv_innovation_covar))
        self.x = self.x_prior + dot(self.kalman_gain,self.innovation)
        self.x_post = deepcopy(self.x)
        self.P = dot(self._identity_mat-dot(self.kalman_gain,self.measurement_mat), self.P_prior)
        self.P_post = deepcopy(self.P)

        return self


    def predict_update(self, z:Union[list,tuple,np.ndarray],
                       u:Union[list,tuple,np.ndarray,type(None)]=None,
                       state_transition_mat:Union[list,tuple,np.ndarray,type(None)]=None,
                       measurement_mat:Union[list,tuple,np.ndarray,type(None)]=None,
                       process_noise_covar:Union[list,tuple,np.ndarray,type(None)]=None,
                       measurement_noise_covar:Union[list,tuple,np.ndarray,type(None)]=None,
                       control_transition_mat:Union[list,tuple,np.ndarray,type(None)]=None):
        """

        """
        return self.predict(
            u=u,
            state_transition_mat=state_transition_mat,
            process_noise_covar=process_noise_covar,
            control_transition_mat=control_transition_mat
        ).update(
            z=z,
            measurement_mat=measurement_mat,
            measurement_noise_covar=measurement_noise_covar
        )


def runge_kutta4(y:Union[int,float], x:Union[int,float], dx:Union[int,float], f:Callable[[Union[int,float],Union[int,float]],Union[int,float]]):
    """computes 4th order Runge-Kutta for dy/dx.
    Parameters
    ----------
    y : scalar
        Initial/current value for y
    x : scalar
        Initial/current value for x
    dx : scalar
        difference in x (e.g. the time step)
    f : ufunc(y,x)
        Callable function (y, x) that you supply to compute dy/dx for
        the specified values.
    """
    k1 = dx * f(y, x)
    k2 = dx * f(y + 0.5*k1, x + 0.5*dx)
    k3 = dx * f(y + 0.5*k2, x + 0.5*dx)
    k4 = dx * f(y + k3, x + dx)

    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.


def mul(mean1:Union[int,float], var1:Union[int,float], mean2:Union[int,float], var2:Union[int,float]) -> Tuple[Union[int,float]]:
    """
    Multiply Gaussian (mean1, var1) with (mean2, var2) and return the
    results as a tuple (mean, var).
    Strictly speaking the product of two Gaussian PDFs is a Gaussian
    function, not Gaussian PDF. It is, however, proportional to a Gaussian
    PDF, so it is safe to treat the output as a PDF for any filter using
    Bayes equation, which normalizes the result anyway.
    Parameters
    ----------
    mean1 : scalar
         mean of first Gaussian
    var1 : scalar
         variance of first Gaussian
    mean2 : scalar
         mean of second Gaussian
    var2 : scalar
         variance of second Gaussian
    Returns
    -------
    mean : scalar
        mean of product
    var : scalar
        variance of product
    Examples
    --------
    >>> mul(1, 2, 3, 4)
    (1.6666666666666667, 1.3333333333333333)
    References
    ----------
    Bromily. "Products and Convolutions of Gaussian Probability Functions",
    Tina Memo No. 2003-003.
    http://www.tina-vision.net/docs/memos/2003-003.pdf
    """
    mean = (var1*mean2 + var2*mean1) / (var1 + var2)
    var = 1 / (1/var1 + 1/var2)
    return (mean, var)


def mul_pdf(mean1:Union[int,float], var1:Union[int,float], mean2:Union[int,float], var2:Union[int,float]) -> Tuple[Union[int,float]]:
    """
    Multiply Gaussian (mean1, var1) with (mean2, var2) and return the
    results as a tuple (mean, var, scale_factor).
    Strictly speaking the product of two Gaussian PDFs is a Gaussian
    function, not Gaussian PDF. It is, however, proportional to a Gaussian
    PDF. `scale_factor` provides this proportionality constant
    Parameters
    ----------
    mean1 : scalar
         mean of first Gaussian
    var1 : scalar
         variance of first Gaussian
    mean2 : scalar
         mean of second Gaussian
    var2 : scalar
         variance of second Gaussian
    Returns
    -------
    mean : scalar
        mean of product
    var : scalar
        variance of product
    scale_factor : scalar
        proportionality constant
    Examples
    --------
    >>> mul(1, 2, 3, 4)
    (1.6666666666666667, 1.3333333333333333)
    References
    ----------
    Bromily. "Products and Convolutions of Gaussian Probability Functions",
    Tina Memo No. 2003-003.
    http://www.tina-vision.net/docs/memos/2003-003.pdf
    """
    mean = (var1*mean2 + var2*mean1) / (var1 + var2)
    var = 1. / (1./var1 + 1./var2)

    S = np.exp(-(mean1-mean2)*(mean1-mean2) / (2*(var1+var2))) / sqrt(2*np.pi*(var1+var2))

    return mean, var, S


def add(mean1:Union[int,float], var1:Union[int,float], mean2:Union[int,float], var2:Union[int,float]) -> Tuple[Union[int,float]]:
    """
    Add the Gaussians (mean1, var1) with (mean2, var2) and return the
    results as a tuple (mean,var).
    var1 and var2 are variances - sigma squared in the usual parlance.
    """
    return (mean1+mean2, var1+var2)


def multivariate_multiply(m1:Union[list,tuple,np.ndarray], c1:Union[list,tuple,np.ndarray], m2:Union[list,tuple,np.ndarray], c2:Union[list,tuple,np.ndarray]) -> Tuple[np.ndarray]:
    """
    Multiplies the two multivariate Gaussians together and returns the
    results as the tuple (mean, covariance).
    Examples
    --------
    .. code-block:: Python
        m, c = multivariate_multiply([7.0, 2], [[1.0, 2.0], [2.0, 1.0]],
                                     [3.2, 0], [[8.0, 1.1], [1.1,8.0]])

    Parameters
    ----------
    m1 : array-like
        Mean of first Gaussian. Must be convertable to an 1D array via np.asarray(),
        For example 6, [6], [6, 5], np.array([3, 4, 5, 6]) are all valid.
    c1 : matrix-like
        Covariance of first Gaussian. Must be convertable to an 2D array via np.asarray().
    m2 : array-like
        Mean of second Gaussian. Must be convertable to an 1D array via np.asarray(),
        For example 6, [6], [6, 5], np.array([3, 4, 5, 6]) are all valid.
    c2 : matrix-like
        Covariance of second Gaussian. Must be convertable to an 2D array via np.asarray()
    
    Returns
    -------
    m : np.ndarray
        mean of the result
    c : np.ndarray
        covariance of the result
    """
    C1 = np.asarray(c1)
    C2 = np.asarray(c2)
    M1 = np.asarray(m1)
    M2 = np.asarray(m2)

    sum_inv = np.linalg.inv(C1+C2)
    C3 = np.dot(C1, sum_inv).dot(C2)

    M3 = (np.dot(C2, sum_inv).dot(M1) + np.dot(C1, sum_inv).dot(M2))

    return M3, C3


def NESS(xs:Union[list,tuple,np.ndarray], est_xs:Union[list,tuple,np.ndarray], ps:Union[list,tuple,np.ndarray]) -> list:
    """
    Computes the normalized estimated error squared test on a sequence
    of estimates. The estimates are optimal if the mean error is zero and
    the covariance matches the Kalman filter's covariance. If this holds,
    then the mean of the NESS should be equal to or less than the dimension
    of x.

    Examples
    --------
    .. code-block: Python
        xs = ground_truth()
        est_xs, ps, _, _ = kf.batch_filter(zs)
        NESS(xs, est_xs, ps)
    
    Parameters
    ----------
    xs : list-like
        sequence of true values for the state x
    est_xs : list-like
        sequence of estimates from an estimator (such as Kalman filter)
    ps : list-like
        sequence of covariance matrices from the estimator
    
    Returns
    -------
    ness : list of floats
       list of NESS computed for each estimate
    """
    est_err = xs - est_xs
    ness = []
    for x, p in zip(est_err, ps):
        ness.append(np.dot(x.T, linalg.inv(p)).dot(x))
    return ness


def inv_diagonal(S:np.ndarray)->np.ndarray:
    """
    Computes the inverse of a diagonal NxN np.array S. In general this will
    be much faster than calling np.linalg.inv().
    
    Parameters
    ----------
    S : np.array
        diagonal NxN array to take inverse of
    
    Returns
    -------
    S_inv : np.array
        inverse of S
    """
    S = np.asarray(S)

    if S.ndim != 2 or S.shape[0] != S.shape[1] or (S-diag(diag(S))!=0).any():
        raise ValueError('S must be a square diagonal matrix')

    si = np.zeros(S.shape)
    for i in range(len(S)):
        si[i, i] = 1. / S[i, i]
    return si


def outer_product_sum(A:np.ndarray, B:Optional[np.ndarray]=None):
    r"""
    Computes the sum of the outer products of the rows in A and B
        P = \Sum {A[i] B[i].T} for i in 0..N
        Notionally:
        P = 0
        for y in A:
            P += np.outer(y, y)
    This is a standard computation for sigma points used in the UKF, ensemble
    Kalman filter, etc., where A would be the residual of the sigma points
    and the filter's state or measurement.
    The computation is vectorized, so it is much faster than the for loop
    for large A.

    Parameters
    ----------
    A : np.array, shape (M, N)
        rows of N-vectors to have the outer product summed
    B : np.array, shape (M, N)
        rows of N-vectors to have the outer product summed
        If it is `None`, it is set to A.
    
    Returns
    -------
    P : np.array, shape(N, N)
        sum of the outer product of the rows of A and B
    
    Examples
    --------
    Here sigmas is of shape (M, N), and x is of shape (N). The two sets of
    code compute the same thing.
    >>> P = outer_product_sum(sigmas - x)
    >>>
    >>> P = 0
    >>> for s in sigmas:
    >>>     y = s - x
    >>>     P += np.outer(y, y)
    """
    if B is None:
        B = A

    outer = np.einsum('ij,ik->ijk', A, B)
    return np.sum(outer, axis=0)
