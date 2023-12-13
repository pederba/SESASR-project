import sympy
sympy.init_printing(use_latex='mathjax')
from sympy import symbols, Matrix, latex

x, y, theta, v, w, dt = symbols('x y theta v w dt')
gux = Matrix([[x + v/w*sympy.sin(theta + w*dt) - v/w*sympy.sin(theta)], 
              [y + v/w*sympy.cos(theta) - v/w*sympy.cos(theta + w*dt)], 
              [theta + w*dt]])

eval_gux = sympy.lambdify((x, y, theta, v, w, dt), gux, 'numpy')

Gt = gux.jacobian(Matrix([x, y, theta]))
eval_Gt = sympy.lambdify((x, y, theta, v, w, dt), Gt, 'numpy')
Vt = gux.jacobian(Matrix([v, w]))
eval_Vt = sympy.lambdify((x, y, theta, v, w, dt), Vt, 'numpy')

