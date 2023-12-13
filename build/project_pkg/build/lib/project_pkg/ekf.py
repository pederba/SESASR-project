import numpy as np

class RobotEKF:
    def __init__(
        self,
        dim_x=1, dim_z=1, dim_u=1,
        eval_gux=None, eval_Gt=None, eval_Vt=None,
        eval_hx=None, eval_Ht=None,
    ):
        """
        Initializes the extended Kalman filter creating the necessary matrices
        """
        self.mu = np.zeros((dim_x, 1))  # mean state estimate
        self.Sigma = np.eye(dim_x)  # covariance state estimate
        self.Mt = np.eye(dim_u)  # process noise
        self.Qt = np.eye(dim_z)  # measurement noise

        self.eval_gux = eval_gux
        self.eval_Gt = eval_Gt
        self.eval_Vt = eval_Vt

        self.eval_hx = eval_hx
        self.eval_Ht = eval_Ht

        self._I = np.eye(dim_x)  # identity matrix used for computations