import sympy
import numpy as np
from sympy import symbols, Matrix

# Velocity model
x, y, theta, v, w, dt = symbols('x y theta v w dt')
gux = Matrix([[x + v/w*sympy.sin(theta + w*dt) - v/w*sympy.sin(theta)], 
              [y + v/w*sympy.cos(theta) - v/w*sympy.cos(theta + w*dt)], 
              [theta + w*dt]])

eval_gux_vel = sympy.lambdify((x, y, theta, v, w, dt), gux, 'numpy')

Gt = gux.jacobian(Matrix([x, y, theta]))
eval_Gt_vel = sympy.lambdify((x, y, theta, v, w, dt), Gt, 'numpy')
Vt = gux.jacobian(Matrix([v, w]))
eval_Vt_vel = sympy.lambdify((x, y, theta, v, w, dt), Vt, 'numpy')

# Velocity motion model for w=0
gux_sing = Matrix([[x + v*dt*sympy.cos(theta)], 
                   [y + v*dt*sympy.sin(theta)], 
                   [theta]])
eval_gux_sing = sympy.lambdify((x, y, theta, v, w, dt), gux_sing, 'numpy')
Gt_sing = gux_sing.jacobian(Matrix([x, y, theta]))
eval_Gt_sing = sympy.lambdify((x, y, theta, v, w, dt), Gt_sing, 'numpy')
Vt_sing = gux_sing.jacobian(Matrix([v, w]))
eval_Vt_sing = sympy.lambdify((x, y, theta, v, w, dt), Vt_sing, 'numpy')

# Odometry motion model
def get_odometry_input(x, x_prev):
    rot1 = np.arctan2(x[1] - x_prev[1], x[0] - x_prev[0]) - x_prev[2]
    trasl = np.sqrt((x[0] - x_prev[0])**2 + (x[1] - x_prev[1])**2)
    rot2 = x[2] - x_prev[2] - rot1
    return np.array([[rot1, trasl, rot2]]).T

x, y, theta, rot1, trasl, rot2 = symbols('x y theta rot1 trasl rot2')

gux_odom = Matrix([[x + trasl*sympy.cos(theta + rot1)],
                   [y + trasl*sympy.sin(theta + rot1)],
                   [theta + rot1 + rot2]])
Gt_odom = gux_odom.jacobian(Matrix([x, y, theta]))
Vt_odom = gux_odom.jacobian(Matrix([rot1, trasl, rot2]))

eval_gux_odom = sympy.lambdify((x, y, theta, rot1, trasl, rot2), gux_odom, 'numpy')
eval_Gt_odom = sympy.lambdify((x, y, theta, rot1, trasl, rot2), Gt_odom, 'numpy')
eval_Vt_odom = sympy.lambdify((x, y, theta, rot1, trasl, rot2), Vt_odom, 'numpy')