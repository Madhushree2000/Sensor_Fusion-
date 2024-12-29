from dataclasses import dataclass

import numpy as np
from numpy import ndarray
from mytypes import MultiVarGauss
from solution import dynamicmodels as dynamicmodels_solu


@dataclass
class WhitenoiseAcceleration2D:
    """
    A white noise acceleration model, also known as constan velocity.
    States are position and speed.
    """
    sigma_a: float  # noise standard deviation

    def f(self, x: ndarray, dt: float,) -> ndarray:
        """Calculate the zero noise transition from x given dt."""
        A = np.array([[0,0,1,0],
             [0,0,0,1],
             [0,0,0,0],
             [0,0,0,0]])
        
        G = np.array([[0,0],
             [0,0],
             [1,0],
             [0,1]])
        
        D = np.array([[self.sigma_a**2 , 0] ,
             [0 , self.sigma_a**2]])
        n = MultiVarGauss(0,D*dt)
        x_next = A @ x + G @ n

        """# TODO replace this with own code
        x_next = dynamicmodels_solu.WhitenoiseAcceleration2D.f(self, x, dt)"""
        return x_next

    def F(self, x: ndarray, dt: float,) -> ndarray:
        """Calculate the discrete transition matrix given dt
        See (4.64) in the book."""

        F = np.array([[1,0,dt,0],
             [0,1,0,dt],
             [0,0,1,0],
             [0,0,0,1]]) 


        """# TODO replace this with own code
        F = dynamicmodels_solu.WhitenoiseAcceleration2D.F(self, x, dt)"""
        return F

    def Q(self, x: ndarray, dt: float,) -> ndarray:
        """Calculate the discrete transition Covariance.
        See(4.64) in the book."""

        a = dt**3 /3
        b = dt**2 / 2

        Q = np.array([[a,0,b,0],
             [0,a,0,b],
             [b,0,dt,0],
             [0,b,0,dt]]) * self.sigma_a**2 
        
        Q = Q 

        """# TODO replace this with own code
        Q = dynamicmodels_solu.WhitenoiseAcceleration2D.Q(self, x, dt)"""
        return Q

    def predict_state(self,
                      state_est: MultiVarGauss,
                      dt: float,
                      ) -> MultiVarGauss:
        """Given the current state estimate, 
        calculate the predicted state estimate.
        See 2. and 3. of Algorithm 1 in the book."""
        x_upd_prev, P = state_est

        F = self.F(x_upd_prev,dt)
        Q = self.Q(x_upd_prev,dt)

        x_pred = F @ x_upd_prev
        P_pred = F@P@F.T + Q

        state_pred_gauss = MultiVarGauss(x_pred, P_pred)

        """# TODO replace this with own code
        state_pred_gauss = dynamicmodels_solu.WhitenoiseAcceleration2D.predict_state(
            self, state_est, dt)"""

        return state_pred_gauss
