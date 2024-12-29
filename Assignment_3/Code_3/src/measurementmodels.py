from dataclasses import dataclass
import numpy as np
from numpy import ndarray
from mytypes import MultiVarGauss
from solution import measurementmodels as measurementmodels_solu


@dataclass
class CartesianPosition2D:
    sigma_z: float

    def h(self, x: ndarray) -> ndarray:
        """Calculate the noise free measurement value of x."""
        x_h = np.array([np.sqrt(x[0]**2+ x[1]**2), np.arctan2(x[1],x[0])]).T

        """# TODO replace this with own code
        x_h = measurementmodels_solu.CartesianPosition2D.h(self, x)"""
        return x_h

    def H(self, x: ndarray) -> ndarray:
        """Get the measurement matrix at x."""

        H = np.array([[1,0,0,0],
                      [0,1,0,0]])

        """ # TODO replace this with own code
        H = measurementmodels_solu.CartesianPosition2D.H(self, x)"""
        return H

    def R(self, x: ndarray) -> ndarray:
        """Get the measurement covariance matrix at x."""

        R = np.array([[self.sigma_z**2 , 0], 
                      [0, self.sigma_z**2]])


        """# TODO replace this with own code
        R = measurementmodels_solu.CartesianPosition2D.R(self, x)"""
        return R

    def predict_measurement(self,
                            state_est: MultiVarGauss
                            ) -> MultiVarGauss:
        """Get the predicted measurement distribution given a state estimate.
        See 4. and 6. of Algorithm 1 in the book.
        """
        x_mean, P = state_est

        H = self.H(x_mean)
        R = self.R(x_mean)

        z_hat = H @ x_mean
        S = H @ P @ H.T + R

        measure_pred_gauss = MultiVarGauss(z_hat, S)

        """# TODO replace this with own code
        measure_pred_gauss = measurementmodels_solu.CartesianPosition2D.predict_measurement(
            self, state_est)"""

        return measure_pred_gauss
