from typing import Tuple
from numpy import ndarray
import numpy as np
from tuning import SimParams


def rotmat2d(theta: float = 0, cos: float = None, sin: float = None):
    ct = cos or np.cos(theta)
    st = sin or np.sin(theta)
    R = np.array(
        [[ct, -st],
         [st, ct]]
    )
    return R


def f_m2_withT(
        x: np.ndarray,
        T: float,
) -> np.ndarray:
    """ CT transition function"""
    if abs(x[4]) > 0.0001:
        xout = x
        theta = T * x[4]
        st = np.sin(theta)
        ct = np.cos(theta)
        xout[0] += st * x[2] / x[4] - (1 - ct) * x[3] / x[4]
        xout[1] += (1 - ct) * x[2] / x[4] + st * x[3] / x[4]
        xout[2:4] = rotmat2d(cos=ct, sin=st) @ xout[2:4]
    else:
        xout = x
        xout[:2] = T * xout[2:4]
    return xout


def sample_CT_trajectory(sim_params: SimParams) -> Tuple[ndarray, ndarray]:
    """
    Sample a state trajectory X with measurements Z for n time steps of size Ts.

    The initial state is sampled from a Gaussian with mean xbar0 and covariance P0. The state
    trasitions follows that of the CV model where the linear acceleration covariance is specified by
    sigma_a and the rotational acceleration has covariance sigma_omegaq. The measurements are
    generated by sampling a Gaussian with mean x[:2] and covarance sigma_z * eye(2).
    """
    xbar0 = np.asfarray(sim_params.x0)
    P = np.asfarray(sim_params.P0)
    assert xbar0.shape[0] == P.shape[0]
    assert xbar0.shape[0] == P.shape[1]

    # some limits
    maxSpeed = 20
    maxTurn = np.pi/4

    # sqrt(covs)
    sigma_a = sim_params.sigma_a
    sigma_z = sim_params.sigma_z
    sigma_omega = sim_params.sigma_omega
    dt = sim_params.dt
    cholR = sigma_z * np.eye(2)

    Q = np.zeros((5, 5))
    Q[:4, :4] = sigma_a ** 2 * np.array(
        [[dt**3 / 3,    0,          dt**2 / 2,	0],
         [0,            dt**3 / 3,  0,    	    dt**2 / 2],
         [dt**2 / 2,    0,          dt,   	    0],
         [0,            dt**2 / 2,  0,          dt]]
    )
    Q[4, 4] = sigma_omega**2 * dt
    cholQ = np.linalg.cholesky(Q)

    # allocate
    Z = np.zeros((sim_params.N_data, 2))
    X = np.zeros((sim_params.N_data, 5))

    # initialize
    X[0] = xbar0 \
        + np.linalg.cholesky(sim_params.P0) @ np.random.normal(size=5)

    for k in range(sim_params.N_data):
        # limit speed
        X[k, 2:4] = np.minimum(maxSpeed, np.maximum(-maxSpeed, X[k, 2:4]))

        # limit turn rate
        X[k, 4] = np.minimum(maxTurn, np.maximum(-maxTurn, X[k, 4]))

        # measurement
        Z[k] = X[k, :2] + cholR @ np.random.normal(size=2)

        # predict
        if k < sim_params.N_data-1:
            X[k+1] = f_m2_withT(X[k], dt) + cholQ @ np.random.normal(size=5)

    return X, Z
