
import numpy as np
from scipy.linalg import expm


class KalmanPKE:
    def __init__(self, x0, P0, Q, R, dt, L, beta_total, beta_i, lambda_i, C):
        # store current state
        self.x_hat = x0.copy()
        self.P = P0.copy()

        # store initial values for reset()
        self._x0_init = x0.copy()
        self._P0_init = P0.copy()

        self.Q = Q
        self.R = R
        self.dt = dt
        self.L = L
        self.beta_total = beta_total
        self.beta_i = beta_i
        self.lambda_i = lambda_i
        self.C = C

    def step_with_measurement(self, measurement, rho):
        """
        One time-step KF:
        Uses time-varying Ad (ρ changes each step)
        """

        # 1) Build continuous A_c matrix and C
        A_c = np.zeros((7, 7))
        A_c[0, 0] = (rho - self.beta_total) / self.L
        A_c[0, 1:] = self.lambda_i
        A_c[1:, 0] = self.beta_i / self.L
        A_c[1:, 1:] = -np.diag(self.lambda_i)

        # 2) Discretize using matrix exponential
        Ad = expm(A_c * self.dt)

        # 3) Predict
        x_pred = Ad @ self.x_hat
        P_pred = Ad @ self.P @ Ad.T + self.Q

        # 4) Update (Correction)

        y_pred = (self.C @ x_pred).reshape(-1)[0]
        nu = measurement - y_pred
        S = self.C @ P_pred @ self.C.T + self.R
        K = (P_pred @ self.C.T) / S  # Kalman gain

        self.x_hat = x_pred + K * nu
        self.P = (np.eye(7) - K @ self.C) @ P_pred

        return self.x_hat[0]  # return estimated power

    def step_no_measurement(self, rho):
        """
        One time-step KF:
        Uses time-varying Ad (rho changes each step)
        No measurement update
        """

        # 1) Build continuous A_c matrix
        A_c = np.zeros((7, 7))
        A_c[0, 0] = (rho - self.beta_total) / self.L
        A_c[0, 1:] = self.lambda_i
        A_c[1:, 0] = self.beta_i / self.L
        A_c[1:, 1:] = -np.diag(self.lambda_i)

        # 2) Discretize using matrix exponential
        Ad = expm(A_c * self.dt)

        # 3) Predict
        x_pred = (Ad @ self.x_hat)
        P_pred = Ad @ self.P @ Ad.T + self.Q

        self.x_hat = x_pred
        self.P = P_pred

        # return estimated power without measurment update
        return self.x_hat[0]


def compute_reactivity(SS1, SS2, RR, No, baseline_reactiviy):
    '''Computes the reactivity (rho) based on the positions of the control rods. Neutron population is used
       as a proxy for temperature feedback. The coefficients in the polynomial are hypothetical and should be 
       replaced with actual values from the reactor physics data.'''

    SS1rho = -0.0401*(SS1**3) + 3.8511*(SS1**2) - 21.994*SS1 + 37.296
    SS2rho = -0.0203*(SS2**3) + 2.0107*(SS2**2) - 15.201*SS2 + 27.103
    RRrho = -0.0029*(RR**3) + 0.2693*(RR**2) - 0.5052*RR + 0.2674

    # Temperature reactivity feedback
    rho_T = 1.1e-6 * No

    # Convert from pcm → dk/k
    rho = (SS1rho + SS2rho + RRrho - rho_T -
           baseline_reactiviy) * 1e-5  # 5586.9
    return rho


def build_continuous_pke():
    """
    1 prompt neutron population + 6 delayed groups
    """
    L = 1e-5  # neutron generation time [s] (example)

    # multiply betas with 0.7
    beta_i = 0.7 * np.array([0.00025, 0.00125, 0.00120,
                             0.00260, 0.00080, 0.00040])
    lambda_i = 0.7 * np.array([0.0124, 0.0305, 0.111,
                               0.301, 1.14, 3.01])

    beta_total = beta_i.sum()

    n_x = 7  # state space dimension
    # This creates the observation matrix C = [1 0 0 0 0 0 0], meaning we only measure the neutron population and not the precursor concentrations.
    C = np.zeros((1, n_x))
    C[0, 0] = 1.0

    return L, beta_total, beta_i, lambda_i, C


def linear_model_predict(est_Kalman):
    '''A simple linear model to predict the other abundant signals based on the estimated neutron population from the KF.'''

    nf2_log = 2.1989E-06 * est_Kalman + 1.072803448
    nf3_pwr = 2.3593E-06 * est_Kalman + 0.179082421
    nf4_flux = 2.19132E-06 * est_Kalman + 0.182602579

    return nf2_log, nf3_pwr, nf4_flux
