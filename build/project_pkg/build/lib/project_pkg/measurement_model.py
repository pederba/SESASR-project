import numpy as np
import sympy
from sympy import symbols, Matrix
from numpy.random import randn # function to get a random number from a Gaussian distribution


# Landmark measurement model
x, y, theta, mx, my = symbols('x y theta mx my')
hx = Matrix([[sympy.sqrt((mx - x)**2 + (my - y)**2)], 
             [sympy.atan2(my - y, mx - x) - theta]])
eval_hx = sympy.lambdify((x, y, theta, mx, my), hx, 'numpy')

Ht = hx.jacobian(Matrix([x, y, theta]))
eval_Ht = sympy.lambdify((x, y, theta, mx, my), Ht, 'numpy')

def z_landmark(x, lmark, std_rng=0.5, std_brg=0.5, max_range=8.0, fov=np.deg2rad(45)):
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
    if z[0, 0] < max_range and abs(z[1, 0])<fov:
        return z + np.array([[randn() * std_rng**2, randn() * std_brg]]).T
        
    return None

def inverse_eval_hx(z, x):
    """
    Compute the inverse of eval_hx function to obtain the landmark position

    Args:
        z [np.array]: the measurement of the landmark
        x [np.array]: the position of the robot with shape (dim_x, 1)

    Returns:
        lmark [np.array]: the position of the landmark in Cartesian coordinates
    """
    x_val = x.flatten()
    z_val = z.flatten()

    # Solve for x, y, theta using sympy
    x, y, theta = symbols('x y theta')
    eq1 = sympy.sqrt((x - x_val[0])**2 + (y - x_val[1])**2) - z_val[0]
    eq2 = sympy.atan2(y - x_val[1], x - x_val[0]) - theta - z_val[1]

    sol = sympy.solve((eq1, eq2), (x, y, theta))
    lmark_polar = np.array([sol[x], sol[y]])
    
    # Convert polar coordinates to Cartesian coordinates
    #lmark_cartesian = np.array([lmark_polar[0] * np.cos(lmark_polar[1]), lmark_polar[0] * np.sin(lmark_polar[1])])
    
    return lmark_polar

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

