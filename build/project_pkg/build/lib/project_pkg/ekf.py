import numpy as np
from numpy.linalg import inv

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

def predict(self, u, g_extra_args=()):
    """
    Update the state prediction using the control input u and compute the relative uncertainty ellipse

    Modified variables:
        self.mu: the state prediction
        self.Sigma: the covariance matrix of the state prediction
    """
    # Update the state prediction evaluating the motion model
    print(u[:,0])
    self.mu = self.eval_gux(*self.mu[:,0], *u[:,0], *g_extra_args)
    
    # Update the covariance matrix of the state prediction, 
    # you need to evaluate the Jacobians Gt and Vt
    Gt = self.eval_Gt(*self.mu[:,0], *u[:,0], *g_extra_args)
    Vt = self.eval_Vt(*self.mu[:,0], *u[:,0], *g_extra_args)
    self.Sigma = Gt @ self.Sigma @ Gt.T + Vt @ self.Mt @ Vt.T

RobotEKF.predict = predict

def update(self, z, lmark, residual=np.subtract):
    """Performs the update innovation of the extended Kalman filter.

    Parameters
    ----------

    z : np.array
        measurement for this step.

    lmark : [x, y] list-like
        Landmark location in cartesian coordinates.

    residual : function (z, z2), optional
        Optional function that computes the residual (difference) between
        the two measurement vectors. If you do not provide this, then the
        built in minus operator will be used. You will normally want to use
        the built in unless your residual computation is nonlinear (for
        example, if they are angles)
    """
    
    # Convert the measurement to a vector if necessary. Needed for the residual computation
    if np.isscalar(z) and self.dim_z == 1:
        z = np.asarray([z], float)

    # Compute the Kalman gain, you need to evaluate the Jacobian Ht
    Ht = self.eval_Ht(*self.mu[:,0], *lmark)
    self.S = Ht @ self.Sigma @ Ht.T + self.Qt
    self.K = self.Sigma @ Ht.T @ inv(self.S)

    # Evaluate the expected measurement and compute the residual, then update the state prediction
    z_hat = self.eval_hx(*self.mu[:,0], *lmark)

     # Evaluate the expected measurement and compute the residual, then update the state prediction
    z_hat = self.eval_hx(*self.mu[:,0], *lmark)
    self.y = np.array(residual(z[:,0], z_hat[:,0]))
    self.y = self.y.reshape(-1, 1)
    self.mu = self.mu + (self.K @ self.y)

    # P = (I-KH)P(I-KH)' + KRK' is more numerically stable and works for non-optimal K vs the equation
    # P = (I-KH)P usually seen in the literature. 
    # Note that I is the identity matrix.
    I_KH = self._I - self.K @ Ht
    self.Sigma = I_KH @ self.Sigma @ I_KH.T + self.K @ self.Qt @ self.K.T

RobotEKF.update = update