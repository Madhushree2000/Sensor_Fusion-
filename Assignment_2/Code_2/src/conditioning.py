import numpy as np
from gaussian import MultiVarGauss2d
from measurement import Measurement2d
from sensor_model import LinearSensorModel2d
from solution import conditioning as conditioning_solu


def get_cond_state(state: MultiVarGauss2d,
                   sens_modl: LinearSensorModel2d,
                   meas: Measurement2d
                   ) -> MultiVarGauss2d:
    
    pred_meas =  np.linalg.inv(sens_modl.H.T @ np.linalg.inv(sens_modl.R) @ sens_modl.H + np.linalg.inv(state.cov))
    kalman_gain =  pred_meas @ sens_modl.H.T @ np.linalg.inv(sens_modl.R)
    innovation =  meas.value - sens_modl.H.T @ state.mean
    cond_mean =  state.mean + kalman_gain @ innovation 
    p,q = np.shape(kalman_gain @ sens_modl.H)
    cond_cov =   (np.eye(p,q) - kalman_gain @ sens_modl.H) @ state.cov

    cond_state = MultiVarGauss2d(cond_mean, cond_cov)

    return cond_state
