"""
This file is explicitely for testing of the crane problem in various ways.
"""
# %% IMPORT STATEMENTS
import casadi as cs
import os
import sys
import json
# import feasible_sqpmethod
from fslp import fslp
import pickle
from datetime import datetime
import pandas as pd
import numpy as np

# dir_path = os.path.dirname(os.path.realpath(__file__))
# execution_folder = os.path.abspath(os.path.join(dir_path, 'testproblems'))
# sys.path.insert(1, execution_folder)

from P2P_timeoptimal_accelred import scara_problem as tp


# %% Helper functions
def store_stats_fpsqp(solver_obj, f_sol, x_sol):
    """
    Solves the interesting stats of the feasible SQP method in a pandas
    data frame. This includes function, gradient, Hessian evaluations and
    also the optimal value
    """
    stats = solver_obj.stats
    iters = stats['iter_count']
    inner_iters = stats['inner_iter']
    acc_iters = stats['accepted_iter']
    n_eval_f = stats['n_eval_f']
    n_eval_g = stats['n_eval_g']
    # n_eval_hess_l = stats['n_eval_hess_l']
    n_eval_jac_g = stats['n_eval_jac_g']
    n_eval_grad_f = stats['n_eval_grad_f']
    success = stats['success']
    slacks_zero_iter = stats['iter_slacks_zero']
    slacks_zero_n_eval_g = stats['n_eval_g_slacks_zero']
    t_wall = stats['t_wall']
    t_wall_zero_slacks = stats['t_wall_zero_slacks']

    f_opt = float(f_sol)
    x_opt = np.array(x_sol).squeeze()

    return [iters, inner_iters, acc_iters, n_eval_f, n_eval_g, n_eval_grad_f, 
            n_eval_jac_g, success, f_opt, x_opt, slacks_zero_iter, slacks_zero_n_eval_g, t_wall, t_wall_zero_slacks]

def store_stats_ipopt(solver, sol):
    """
    Solves the interesting stats of the casadi SQP method in a pandas
    data frame. This includes function, gradient, Hessian evaluations and
    also the optimal value
    """
    stats = solver.stats()
    iters = stats['iter_count']
    n_eval_f = stats['n_call_nlp_f']
    n_eval_g = stats['n_call_nlp_g']
    # n_eval_hess_l = stats['n_call_nlp_hess_l']
    n_eval_jac_g = stats['n_call_nlp_jac_g']
    n_eval_grad_f = stats['n_call_nlp_grad_f']
    success = stats['success']
    obj_values = stats['iterations']['obj']
    inf_pr = stats['iterations']['inf_pr']
    t_wall_total = stats['t_wall_total']

    f_opt = float(sol['f'])
    x_opt = np.array(sol['x']).squeeze()

    return [iters, n_eval_f, n_eval_g, n_eval_grad_f, n_eval_jac_g,
            success, f_opt, x_opt, obj_values, inf_pr, t_wall_total]


# %% 
# start_start_index = 0
# start_final_index = 10
with open('start_perturbations_scara.pkl', 'rb') as fp:
    start_list_tuples = pickle.load(fp)

# start_list_tuples = start_list_tuples[start_start_index:start_final_index]
start_points = ["start"+str(i) for i in range(len(start_list_tuples))]

# Reload the end list from file
# end_start_index = 0
# end_final_index = 10
with open ('end_perturbations_scara.pkl', 'rb') as fp:
    end_list_tuples = pickle.load(fp)
# end_list_tuples = end_list_tuples[end_start_index:end_final_index]
end_points = ["end"+str(i) for i in range(len(end_list_tuples))]

# Create Multiindeces
mult_ind = pd.MultiIndex.from_product([start_points, end_points],
                                       names=["Startpoints", "Endpoints"])
columns_fpsqp = ['iters', 'inner_iters', 'accepted_iters', 'n_eval_f', 
               'n_eval_g', 'n_eval_grad_f', 'n_eval_jac_g', 
               'success', 'f_opt', 'x_opt', 'slack_zero_iter', 'slack_zero_n_eval_g', 't_wall', 't_wall_zero_slacks'] 

columns_ipopt = ['iters', 'n_eval_f', 'n_eval_g', 'n_eval_grad_f', 
                'n_eval_jac_g', 'success', 'f_opt', 'x_opt', 'obj', 'inf_pr', 't_wall_total']

problem_stats_ipopt = []
problem_stats_fpsqp = []

# %% Define options
#         solve_str = 'qpoases'

# if solve_str == 'ipopt':
# with open(os.path.join(dir_path,
#                     'opts_qpsolver_ipopt.txt'), 'r') as file:
#     opts = json.loads(file.read())
# opts = {}
# opts['qpsol'] = 'qpoases'
# opts['qpsol_opts'] = {'nWSR':10000, "printLevel": "none", "sparse":True, 'hessian_type': 'semidef'}
opts = {}
opts['qpsol'] = 'cplex'
opts['qpsol_opts'] = {'verbose':False, 'tol':1e-9, 'qp_method':2, 'warm_start':True, 'dep_check':0, 'cplex':{'CPXPARAM_Simplex_Display':0, 'CPXPARAM_ScreenOutput':0}}
max_iter = 200
max_inner_iter = 100
contraction_acceptance = 0.3
watchdog = 5
feas_tol = 1e-7
tr_eta1 = 0.25
tr_eta2 = 0.75
tr_alpha1 = 0.25
tr_alpha2 = 2
tr_tol = 1e-8

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
opts['opt_check_slacks'] = True
opts['n_slacks_start'] = 4
opts['n_slacks_end'] = 4
# opts['gradient_correction'] = True


# %% Solve the problem
for i in range(len(start_list_tuples)):
    for j in range(len(end_list_tuples)):

        testproblem = tp(N=20)
        (x, f, g, lbg, ubg, lbx, ubx, x0) = testproblem.create_problem(dev_lx_0=start_list_tuples[i], dev_lx_f=end_list_tuples[j])
        opts['testproblem_obj'] = testproblem

        # # SOLVE NLP WITH IPOPT
        nlp = {'x': x, 'f': f, 'g': g}
        opts_ipopt = {'print_time':True, 'ipopt':{'print_level':5, 'sb':'yes', 'linear_solver':'ma57'}}
        ipopt_solver = cs.nlpsol('ipopt_solver', 'ipopt', nlp, opts_ipopt)

        sol = ipopt_solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        stats_ipopt = store_stats_ipopt(ipopt_solver, sol)
        problem_stats_ipopt.append(stats_ipopt)
        
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
        init_dict['tr_rad0'] = 1.0

        init_dict['tr_scale_mat0'], init_dict['tr_scale_mat_inv0'] = testproblem.create_scaling_matrices()
        
        feasible_solver = fslp.FSLP_Method()
        x_sol, f_sol = feasible_solver.solve(problem_dict, init_dict, opts)

        stats_fpsqp = store_stats_fpsqp(feasible_solver, f_sol, x_sol)
        problem_stats_fpsqp.append(stats_fpsqp)

# %% Storing of the results
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y")

df_stats_ipopt = pd.DataFrame(problem_stats_ipopt, columns=columns_ipopt, index=mult_ind)
df_stats_ipopt.to_pickle('feasible_ipopt_stats_crane_FE100_MA57'+dt_string+'.pkl')
pd.set_option("max_rows", None)
print(df_stats_ipopt)

df_stats_fpsqp = pd.DataFrame(problem_stats_fpsqp, columns=columns_fpsqp, index=mult_ind)
df_stats_fpsqp.attrs['max_iter'] = max_iter
df_stats_fpsqp.attrs['max_inner_iter'] = max_inner_iter
df_stats_fpsqp.attrs['contraction_acceptance'] = contraction_acceptance
df_stats_fpsqp.attrs['watchdog'] = watchdog
df_stats_fpsqp.attrs['feas_tol'] = feas_tol
df_stats_fpsqp.attrs['tr_eta1'] = tr_eta1
df_stats_fpsqp.attrs['tr_eta2'] = tr_eta2
df_stats_fpsqp.attrs['tr_alpha1'] = tr_alpha1
df_stats_fpsqp.attrs['tr_alpha2'] = tr_alpha2
df_stats_fpsqp.attrs['tr_tol'] = tr_tol
# df_stats_fpsqp.attrs['start_start_index']= start_start_index
# df_stats_fpsqp.attrs['start_final_index']= start_final_index
# df_stats_fpsqp.attrs['end_start_index']= end_start_index
# df_stats_fpsqp.attrs['end_final_index']= end_final_index
df_stats_fpsqp.to_pickle('feasible_slp_stats_scara_wCPLEX_whole_dataset'+dt_string+'.pkl')


pd.set_option("max_rows", None)
print(df_stats_fpsqp)
