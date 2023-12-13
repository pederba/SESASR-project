import numpy as np
import math
import sympy
from project_pkg.motion_models import eval_hx
from numpy.random import randn # function to get a random number from a Gaussian distribution


def z_landmark(x, lmark, std_rng=0.5, std_brg=0.5):
    """
    Simulate the measurement of a landmark adding some Gaussian noise

    Args:
        x [np.array]: the position of the robot with shape (dim_x, 1)
        lmark [np.array or List]: the position of the landmark, eg. [x, y]
        std_rng [float]: the standard deviation of the range measurement
        std_brg [float]: the standard deviation of the bearing measurement
    Returns:
        z [np.array]: the measurement of the landmark
    """
    z = eval_hx(*x, *lmark)

    # filter z for a more realistic sensor simulation (add a max range distance and a FOV)
    fov = np.deg2rad(45)
    if z[0, 0] < 8.0 and abs(z[1, 0])<fov:
        return z + np.array([[randn() * std_rng**2, randn() * std_brg]]).T
        
    return None

def residual(a, b):
    """
    Compute the residual between measuremnts, normalizing angles between [-pi, pi)

    Returns:
        y [np.array] : the residual between the two states
    """
    
    y = a - b
    y[1] = y[1] % (2 * np.pi)  # force in range [0, 2 pi)
    if y[1] > np.pi:  # move to [-pi, pi)
        y[1] -= 2 * np.pi
    return y

