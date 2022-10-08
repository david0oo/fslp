# %% IMPORT STATEMENTS
import casadi as cs
import os
import sys
import json
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from src.fslp import fslp
import pandas as pd
import pickle


from examples.CDC_2022_Results.P2P_timeoptimal_crane_problem import crane_problem as tp
# from P2P_fixedtime_crane_problem import crane_problem as tp

dir_path = os.path.dirname(os.path.realpath(__file__))
# execution_folder = os.path.abspath(os.path.join(dir_path, 'examples', 'CDC_2022_results'))
# sys.path.insert(1, execution_folder)

# %% CREATE TEST PROBLEM
# Load list of starting points
with open (dir_path+'/examples/CDC_2022_Results/start_perturbations_crane.pkl', 'rb') as fp:
    start_list_tuples = pickle.load(fp)

start_list_tuple = start_list_tuples[0]
# Load list of end points 
with open (dir_path+'/examples/CDC_2022_Results/end_perturbations_crane.pkl', 'rb') as fp:
    end_list_tuples = pickle.load(fp)

end_list_tuple = end_list_tuples[4]

testproblem = tp(N=20)
# (x, f, g, lbg, ubg, lbx, ubx, x0) = testproblem.create_problem(dev_lx_0=start_list_tuple, dev_lx_f=end_list_tuple)
(x, f, g, lbg, ubg, lbx, ubx, x0) = testproblem.create_problem()


# %% SOLVE NLP WITH IPOPT
nlp = {'x': x, 'f': f, 'g': g}
opts_ipopt = {'ipopt':{'fixed_variable_treatment': 'make_constraint', 'linear_solver':'ma57'}}
opts_sqpmethod = {'convexify_strategy':'regularize', 'max_iter':200}
ipopt_solver = cs.nlpsol('ipopt_solver', 'ipopt', nlp, opts_ipopt)
sol = ipopt_solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
# # The optimal solution of the NLP..This solution should be found
x_opt_ipopt = sol['x']
lam_g_opt_ipopt = sol['lam_g']

# %% CREATE PROBLEM AND INIT DICTS FOR FP-SQP SOLVER
problem_dict = {}
problem_dict['x'] = x
problem_dict['f'] = f
problem_dict['g'] = g

init_dict = {}
init_dict['lbx'] = lbx
init_dict['ubx'] = ubx
init_dict['lbg'] = lbg
init_dict['ubg'] = ubg
init_dict['x0'] = x0
# init_dict['lam_g0'] = lam_g_0
init_dict['tr_rad0'] = 1.0

# Time optimal crane unslacked state constraints
init_dict['tr_scale_mat0'], init_dict['tr_scale_mat_inv0'] = testproblem.create_scaling_matrices()

# %% DEFINE OPTIONS DICT
solve_str = 'cplex'

if solve_str == 'ipopt':
    with open(os.path.join(dir_path,
                        'opts_qpsolver_ipopt.txt'), 'r') as file:
        opts = json.loads(file.read())
elif solve_str == 'qpoases':
    opts = {}
    opts['lpsol'] = 'qpoases'
    # # # opts['lpsol_opts'] = {'nWSR':5000, "schur":True, "printLevel": "none", "enableEqualities":False, "linsol_plugin":"ma27", "max_schur":200}
    # opts['lpsol_opts'] = {'nWSR':10000, "sparse":True, 'hessian_type': 'semidef'}
    # opts['lpsol_opts'] = {'nWSR':10000, "printLevel": "none", "sparse":True, 'hessian_type': 'semidef'}
elif solve_str == 'clp':
    opts = {}
    opts['lpsol'] = 'clp'
    opts['lpsol_opts'] = {'verbose':False, 'SolveType':'useDual', 'clp':{'PrimalTolerance':1e-12, 'DualTolerance':1e-10}}
elif solve_str == 'gurobi':
    opts = {}
    opts['lpsol'] = 'gurobi'
    opts['lpsol_opts'] = {'verbose':False, 'clp':{'PrimalTolerance':1e-12, 'DualTolerance':1e-10}}
elif solve_str == 'cplex':
    opts = {}
    opts['lpsol'] = 'cplex'
    opts['lpsol_opts'] = {'verbose':False, 'tol':1e-9, 'qp_method':2, 'warm_start':True, 'dep_check':2, 'cplex':{'CPXPARAM_Simplex_Display':0, 'CPXPARAM_ScreenOutput':0}}#, 'cplex':{'tol':1e-12}}
else:
    raise ValueError('Wrong string given!')

opts['max_iter'] = 100
opts['optim_tol'] = 1e-7
opts['max_inner_iter'] = 40
opts['tr_eta1'] = 0.25
opts['tr_eta2'] = 0.75
opts['tr_alpha1'] = 0.25
opts['tr_alpha2'] = 2
opts['tr_tol'] = 1e-8
opts['opt_check_slacks'] = True
# opts['verbose'] = False
opts['testproblem_obj'] = testproblem
opts['n_slacks_start'] = 6
opts['n_slacks_end'] = 6
# opts['gradient_correction'] = True
# opts['single_shooting_correction'] = True

# %% SOLVE THE PROBLEM WITH FP-SQP

# tols = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
tols = [1e-5]
opts['feas_tol'] = 1e-7
watchdogs = [5]#[8, 7, 6, 5, 4, 3, 2]
contraction_acc = [0.3]#[0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.1, 0.01]

n_acc_iters = {}
n_acc_iters['contraction_acc'] = contraction_acc
n_inner_iters = {}
n_inner_iters['contraction_acc'] = contraction_acc
success_stories = {}
success_stories['contraction_acc'] = contraction_acc
for j in range(len(watchdogs)):
    n_acc_iters_in = []
    n_inner_iters_in = []
    successes = []
    for i in range(len(contraction_acc)):
        opts['contraction_acceptance'] = contraction_acc[i]
        opts['watchdog'] = watchdogs[j]
        feasible_solver = fslp.FSLP_Method()
        x_sol, f_sol = feasible_solver.solve(problem_dict, init_dict, opts)
        n_inner_iters_in.append(feasible_solver.inner_iter_count)
        n_acc_iters_in.append(feasible_solver.accepted_counter)
        successes.append(feasible_solver.success)
    n_inner_iters[str(watchdogs[j])] = n_inner_iters_in
    n_acc_iters[str(watchdogs[j])] = n_acc_iters_in
    success_stories[str(watchdogs[j])] = successes

# %% DO POSTPROCESSING OF SOLUTIONS
df_inner_iters = pd.DataFrame(data=n_inner_iters)
print('Inner iterates:\n', df_inner_iters)
df_acc_iters = pd.DataFrame(data=n_acc_iters)
print('Accepted iterates:\n', df_acc_iters)
df_success = pd.DataFrame(data=success_stories)
print('Success Stories:\n', df_success)

print('SQP eval f:', feasible_solver.stats['n_eval_f'])
print('SQP eval g:', feasible_solver.stats['n_eval_g'])
print('SQP eval grad f:', feasible_solver.stats['n_eval_grad_f'])
print('SQP eval jac g:', feasible_solver.stats['n_eval_jac_g'])
# print('SQP eval hess l:', feasible_solver.stats['n_eval_hess_l'])
print('SQP iter:', feasible_solver.stats['iter_count'])
print('SQP accepted iter:', feasible_solver.stats['accepted_iter'])
print('SQP inner iter:', feasible_solver.stats['inner_iter'])
print('SQP wall clock time: ', feasible_solver.stats['t_wall'])
print('SQP wall clock time slacks zero: ', feasible_solver.stats['t_wall_zero_slacks'])


# print("Tols: ", tols)
# print("Watchdogs: ", watchdogs)
# print("Contraction_acc: ", contraction_acc)
# print("Accepted iters: ", n_acc_iters)
# print("Inner iters:", n_inner_iters)


testproblem.plot([x_sol, x_opt_ipopt], ['FSLP', 'IPOPT'])

print("Optimal time FP-SQP: ", testproblem.get_optimal_time(x_sol))
print("Optimal time IPOPT: ", testproblem.get_optimal_time(x_opt_ipopt))
# print('m_ks: ', feasible_solver.list_mks)

# plt.figure()
# iters = np.arange(0, len(feasible_solver.list_mks))
# iter_feas = feasible_solver.list_mks
# plt.semilogy(iters, iter_feas, label='$|m_k|$')
# iters = np.arange(0, len(feasible_solver.list_grad_lag))
# iter_feas = feasible_solver.list_grad_lag
# plt.semilogy(iters, iter_feas, label='Inf-norm Gradient of Lagrangian')
# plt.xlabel('Iterations')
# plt.ylabel('Inf-norm $m_k$')
# plt.title('Optimality Improvement')
# plt.legend()
# plt.show()

fig, ax = plt.subplots()
ax.set_aspect('equal')
rect = Rectangle((0.1, -2), 0.1, 1.3, linewidth=1,
                    edgecolor='r', facecolor='none')
ax.add_patch(rect)
for i in range(len(feasible_solver.list_iter)):
    (_, _, _, p_load) = testproblem.get_particular_states(feasible_solver.list_iter[i])
    ax.plot(p_load[:,0], p_load[:,1], label='iter=%s' % (str(i)))
ax.set_ylim((-1, -0.2))
ax.set_xlim((-0.1, 0.8))
plt.legend(loc='upper right')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Overhead Crane, P2P Motion with obstacle')
plt.show()