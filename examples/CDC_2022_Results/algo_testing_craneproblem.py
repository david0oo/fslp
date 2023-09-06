"""
This file is explicitely for testing of the crane problem in various ways.
"""
# %% IMPORT STATEMENTS
import casadi as cs
import os
import sys
import json
from fslp import fslp
import pickle
from datetime import datetime
import pandas as pd
import numpy as np

# dir_path = os.path.dirname(os.path.realpath(__file__))
# execution_folder = os.path.abspath(os.path.join(dir_path, 'testproblems'))
# sys.path.insert(1, execution_folder)

from P2P_timeoptimal_crane_problem import crane_problem as tp


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
start_final_index = 1#10
with open ('start_perturbations_crane.pkl', 'rb') as fp:
    start_list_tuples = pickle.load(fp)

start_list_tuples = start_list_tuples[0:start_final_index]
start_points = ["start"+str(i) for i in range(len(start_list_tuples))]

# Reload the end list from file
end_final_index = 1#10
with open ('end_perturbations_crane.pkl', 'rb') as fp:
    end_list_tuples = pickle.load(fp)
end_list_tuples = end_list_tuples[0:end_final_index]
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
test_ipopt = False
test_fslp = True
opts = {}
opts['subproblem_solver'] = 'cplex'
opts['subproblem_solver_opts'] = {'error_on_fail':False, 'verbose':False, 'tol':1e-9, 'qp_method':2, 'warm_start':True, 'dep_check':2, 'cplex':{'CPXPARAM_Simplex_Display':0, 'CPXPARAM_ScreenOutput':0}}
max_iter = 200
max_inner_iter = 50
contraction_acceptance = 0.3
watchdog = 5
feas_tol = 1e-7
tr_eta1 = 0.25#0.01
tr_eta2 = 0.75#0.9
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
opts['n_slacks_start'] = 6
opts['n_slacks_end'] = 6

opts_ipopt = {'print_time': True,
              'ipopt': {'print_level':0, 'sb':'yes',
                        'linear_solver': 'ma57'}}

# %% Solve the problem
for i in range(len(start_list_tuples)):
    for j in range(len(end_list_tuples)):

        print('Start index: ', i, 'End index', j)
        testproblem = tp(N=20)
        (x, f, g, lbg, ubg, lbx, ubx, x0) = testproblem.create_problem(
            dev_lx_0=start_list_tuples[i], dev_lx_f=end_list_tuples[j])


        # # SOLVE NLP WITH IPOPT
        if test_ipopt:
            nlp = {'x': x, 'f': f, 'g': g}
            
            ipopt_solver = cs.nlpsol('ipopt_solver', 'ipopt', nlp, opts_ipopt)

            sol = ipopt_solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            # The optimal solution of the NLP..This solution should be found
            stats_ipopt = store_stats_ipopt(ipopt_solver, sol)
            
            problem_stats_ipopt.append(stats_ipopt)
        
        if test_fslp:
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

            init_dict['tr_scale_mat0'], init_dict['tr_scale_mat_inv0'] = testproblem.create_scaling_matrices_python()
            
            feasible_solver = fslp.FSLP(problem_dict, opts)
            x_sol, f_sol, lam_g_sol, lam_x_sol = feasible_solver(init_dict)

            stats_fpsqp = store_stats_fpsqp(feasible_solver, f_sol, x_sol)
            problem_stats_fpsqp.append(stats_fpsqp)

# %% Storing of the results
now = datetime.now()
pd.set_option("max_rows", None)
dt_string = now.strftime("%d-%m-%Y")

# df_stats_ipopt = pd.DataFrame(problem_stats_ipopt, columns=columns_ipopt, index=mult_ind)
# df_stats_ipopt.to_pickle('feasible_ipopt_stats_crane_FE100_'+dt_string+'.pkl')
# print(df_stats_ipopt)


df_stats_fpsqp = pd.DataFrame(problem_stats_fpsqp, columns=columns_fpsqp, index=mult_ind)
df_stats_fpsqp.to_pickle('fslp_stats_crane_wANDERSONm5_'+dt_string+'.pkl')
print(df_stats_fpsqp)

