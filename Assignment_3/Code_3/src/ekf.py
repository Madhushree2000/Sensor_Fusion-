from solution import ekf as ekf_solu
"""
Notation:
----------
x is generally used for either the state or the mean of a gaussian. It should be clear from context which it is.
P is used about the state covariance
z is a single measurement
Z are multiple measurements so that z = Z[k] at a given time step k
v is the innovation z - h(x)
S is the innovation covariance
"""
from dataclasses import dataclass
import numpy as np

from dynamicmodels import WhitenoiseAcceleration2D
from measurementmodels import CartesianPosition2D
from mytypes import MultiVarGauss, Measurement2d


@dataclass
class ExtendedKalmanFilter:
    dyn_modl: WhitenoiseAcceleration2D
    sens_modl: CartesianPosition2D

    def step(self,
             state_old: MultiVarGauss,
             meas: Measurement2d,
             ) -> MultiVarGauss:
        """Given previous state estimate and measurement, 
        return new state estimate.

        Relationship between variable names and equations in the book:
        \hat{x}_{k|k_1} = pres_state.mean
        P_{k|k_1} = pres_state.cov
        \hat{z}_{k|k-1} = pred_meas.mean
        \hat{S}_k = pred_meas.cov
        \hat{x}_k = upd_state_est.mean
        P_k = upd_state_est.cov
        """
        state_pred = self.dyn_modl.predict_state(state_old, meas.dt)
        meas_pred = self.sens_modl.predict_measurement(state_pred)

        H = self.sens_modl.H(state_pred.mean)
        kalman_gain = state_pred.cov @ H.T @ np.linalg.inv(meas_pred.cov)
        innovation =  meas.value - meas_pred.mean

        state_upd_mean = state_pred.mean + kalman_gain @ innovation
        shape =np.shape(kalman_gain @ H)
        state_upd_cov = (np.eye(shape[0], shape[1]) - kalman_gain @ H)@ state_pred.cov

        state_upd = MultiVarGauss(state_upd_mean, state_upd_cov)

        """# TODO replace this with own code
        state_upd, state_pred, meas_pred = ekf_solu.ExtendedKalmanFilter.step(
            self, state_old, meas)"""

        return state_upd, state_pred, meas_pred
