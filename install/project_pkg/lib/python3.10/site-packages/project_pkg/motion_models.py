import sympy
import numpy as np
from sympy import symbols, Matrix


x, y, theta, v, w, dt = symbols('x y theta v w dt')
gux = Matrix([[x + v/w*sympy.sin(theta + w*dt) - v/w*sympy.sin(theta)], 
              [y + v/w*sympy.cos(theta) - v/w*sympy.cos(theta + w*dt)], 
              [theta + w*dt]])

eval_gux = sympy.lambdify((x, y, theta, v, w, dt), gux, 'numpy')

Gt = gux.jacobian(Matrix([x, y, theta]))
eval_Gt = sympy.lambdify((x, y, theta, v, w, dt), Gt, 'numpy')
Vt = gux.jacobian(Matrix([v, w]))
eval_Vt = sympy.lambdify((x, y, theta, v, w, dt), Vt, 'numpy')

mx, my = symbols('mx my')
hx = Matrix([[sympy.sqrt((mx - x)**2 + (my - y)**2)], [sympy.atan2(my - y, mx - x) - theta]])
eval_hx = sympy.lambdify((x, y, theta, mx, my), hx, 'numpy')

Ht = hx.jacobian(Matrix([x, y, theta]))
eval_Ht = sympy.lambdify((x, y, theta, mx, my), Ht, 'numpy')

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