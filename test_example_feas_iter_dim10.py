import casadi as cs
import numpy as np
from matplotlib import pyplot as plt
from fslp import fslp

dim = 2
x = cs.MX.sym('x', dim)
f = x[0] - x[1]#cs.sum1(x[1:])# x[1] - x[2] - x[3] - x[4] - x[5] - x[6] - x[7] - x[8] - x[9]
# f = x[0]**2 + (x[1]-2)**2
g = x[0]**10 + x[1]**10#x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2
lbg = cs.vertcat(2**10)
ubg = cs.vertcat(2**10)
x0 = cs.vertcat(2, 0)

problem_dict = {}
problem_dict['x'] = x
problem_dict['f'] = f
problem_dict['g'] = g

init_dict = {}
init_dict['lbg'] = lbg
init_dict['ubg'] = ubg
init_dict['x0'] = x0
init_dict['tr_rad0'] = 1.0

max_iter = 1
max_inner_iter = 50
contraction_acceptance = 0.3
watchdog = 5
feas_tol = 1e-7
tr_eta1 = 0.25
tr_eta2 = 0.75
tr_alpha1 = 0.25
tr_alpha2 = 2
tr_tol = 1e-8


opts = {}
opts['lpsol'] = 'cplex'
opts['lpsol_opts'] = {'verbose':False, 'tol':1e-9, 'qp_method':2, 'warm_start':True, 'dep_check':0, 'cplex':{'CPXPARAM_Simplex_Display':0, 'CPXPARAM_ScreenOutput':0}, 'jit':True}
opts['max_iter'] = max_iter
opts['max_inner_iter'] = max_inner_iter
opts['contraction_acceptance'] = contraction_acceptance
opts['watchdog'] = watchdog
opts['feas_tol'] = feas_tol
opts['verbose'] = True
opts['tr_eta1'] = tr_eta1
opts['tr_eta2'] = tr_eta2
opts['tr_alpha1'] = tr_alpha1
opts['tr_alpha2'] = tr_alpha2
opts['tr_tol'] = tr_tol
opts['opt_check_slacks'] = False

feasible_solver = fslp.FSLP_Method()
x_sol, f_sol = feasible_solver.solve(problem_dict, init_dict, opts)

print(x_sol)

list_x = [a[0] for a in  feasible_solver.list_inner_iterates[0]]
list_x = cs.vertcat(*list_x)
list_y = [a[1] for a in  feasible_solver.list_inner_iterates[0]]
list_y = cs.vertcat(*list_y)

# Plot the solution here
x_points = np.linspace(1, 2, 50)
y_points1 = np.sqrt((x_points-2)*(-x_points))
y_points2 = -2*np.sqrt((x_points-2)*(-x_points))
circle1 = plt.Circle((0, 0), 2, edgecolor='r', fill=False)
rect1 = plt.Rectangle((1, -1), 2, 2, edgecolor='b', fill=False)
fig, ax = plt.subplots()
ax.add_patch(circle1)
ax.add_patch(rect1)
ax.set_aspect('equal')
plt.vlines(2, -3, 3)
# plt.plot(x_points, y_points1, color='g')
# plt.plot(x_points, y_points2, color='g')
plt.plot(list_x, list_y, color='m', marker='o')
plt.xlim((1,2.1))
plt.ylim((0,1.1))
# plt.show()
print(feasible_solver.list_inner_iterates[0])
