import numpy as np

from mytypes import Measurement2d, MultiVarGauss
from tuning import EKFParams
from solution import initialize as initialize_solu


def get_init_CV_state(meas0: Measurement2d, meas1: Measurement2d,
                      ekf_params: EKFParams) -> MultiVarGauss:
    """This function will estimate the initial state and covariance from
    the two first measurements"""
    dt = meas1.dt
    z0, z1 = meas0.value, meas1.value
    sigma_a = ekf_params.sigma_a
    sigma_z = ekf_params.sigma_z
    K = np.concatenate([np.concatenate([np.eye(2),np.zeros((2,2))],axis=1),np.concatenate([np.eye(2)/dt, - np.eye(2)/dt],axis=1)])
    z = np.concatenate([z1,z0])

    Q = np.eye(2)*sigma_a**2
    R = np.eye(2)*sigma_z**2
    P = np.concatenate([np.concatenate([R,np.zeros((2,2))],axis = 1),np.concatenate([np.zeros((2,2)),R],axis=1)])
    mean = K @ z
    cov = K@P@K.T
    init_state = MultiVarGauss(mean, cov)

    """# TODO replace this with own code
    init_state = initialize_solu.get_init_CV_state(meas0, meas1, ekf_params)"""
    return init_state
