import casadi as cs
import numpy as np
from src.fslp import fslp

from examples.P2P_timeoptimal_crane_problem import crane_problem as tp

testproblem = tp(N=20)
(x, f, g, lbg, ubg, lbx, ubx, x0) = testproblem.create_problem()


# %% SOLVE NLP WITH CASADI FEASIBLE_SQP_METHOD
nlp = {'x': x, 'f': f, 'g': g}


opts_sqpmethod = {  'solve_type': 'SLP',
                        'qpsol': 'cplex',
                        'qpsol_options': {'dump_to_file':True, 'error_on_fail': False, 'verbose':False, 'tol':1e-9, 'qp_method':2, 'warm_start':True, 'dep_check':2, 'cplex':{'CPXPARAM_Simplex_Display':2, 'CPXPARAM_ScreenOutput':1}},
                        # 'qpsol_options': {'osqp':{'verbose':True}},
                        # 'print_time': False,
                        # 'convexify_strategy':'regularize'
                        # 'use_anderson': True,
                        # 'anderson_memory':1
                        }


opts_sqpmethod['max_iter']=1
opts_sqpmethod['max_inner_iter']=0
opts_sqpmethod['tr_rad0']= 1.0
opts_sqpmethod['feas_tol']= 1e-8
opts_sqpmethod['optim_tol']= 1e-8
opts_sqpmethod['hess_lag']= testproblem.create_gn_hessian_cpp()
#opts_sqpmethod[# 'hessian_approximation']= 'limited-memory',
#opts_sqpmethod[# 'lbfgs_memory']= 3,
opts_sqpmethod['tr_scale_vector']= testproblem.create_scaling_matrices_cpp()
# opts_sqpmethod['print_time'] = False

solver = cs.nlpsol("solver", "ipopt", nlp)
# solver = cs.nlpsol("S", "feasiblesqpmethod", nlp, opts_sqpmethod)

# # Solve the problem
# res = solver(x0  = x0, ubg = ubg, lbg = lbg, lbx = lbx, ubx=ubx)
# x_opt_ipopt = res['x']

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
init_dict['tr_scale_mat0'], init_dict['tr_scale_mat_inv0'] = testproblem.create_scaling_matrices_python()

# %% CREATE OPTIONS
opts = {}

# opts['solver_type'] = 'SQP'
# opts['subproblem_sol'] = 'qpoases'
# opts['subproblem_sol_opts'] = {'nWSR':1000000, "printLevel": "none", "sparse":True, 'hessian_type': 'semidef'}#, 'enableEqualities':True}

solve_str = 'ipopt'

# Create an NLP solver
if solve_str == 'ipopt':
    opts = {  'subproblem_sol': 'nlpsol',
                    'subproblem_sol_opts': {  "nlpsol": "ipopt", 
                                        "verbose": True, 
                                        "print_time": False, 
                                        "nlpsol_options": {"ipopt": {   "print_level": 0, 
                                                                        "sb": "yes", 
                                                                        "fixed_variable_treatment": "make_constraint", 
                                                                        "hessian_constant": "yes", 
                                                                        "jac_c_constant": "yes", 
                                                                        "jac_d_constant": "yes",
                                                                        "tol": 1e-12, 
                                                                        "tiny_step_tol": 1e-20, 
                                                                        "mumps_scaling": 0, 
                                                                        "honor_original_bounds": "no", 
                                                                        "bound_relax_factor": 0}, 
                                        "print_time": False}, 
                                        "error_on_fail": False}}
elif solve_str == 'qpoases':
    opts = {  'solve_type': 'SQP',
                        'subproblem_sol': 'qpoases',
                        'subproblem_sol_opts': {  'nWSR':1000000,
                                            # 'schur': True,
                                            # 'linsol_plugin': 'ma57',
                                            'printLevel': 'none',
                                            'sparse':True,
                                            'hessian_type': 'semidef'
                                            # 'enableEqualities':True
                        }}

elif solve_str == 'cplex':

    opts['subproblem_sol'] = 'cplex'
    opts['subproblem_sol_opts'] = {'dump_to_file':False, 'error_on_fail': False, 'verbose':False, 'tol':1e-9, 'qp_method':2, 'warm_start':True, 'dep_check':2, 'cplex':{'CPXPARAM_Simplex_Display':1, 'CPXPARAM_ScreenOutput':1, 'CPXPARAM_ParamDisplay':True, 'CPXPARAM_Read_DataCheck':1, 'CPXPARAM_Conflict_Display':2, 'CPXPARAM_Tune_Display':3, 'CPXPARAM_Preprocessing_Dual':1}}#, 'cplex':{'tol':1e-12}}
    opts['solver_type'] = 'SLP'

opts['max_iter'] = 20
opts['feas_strategy'] = 2
opts['optim_tol'] = 1e-8
opts['max_inner_iter'] = 50
opts['tr_eta1'] = 0.25
opts['tr_eta2'] = 0.75
opts['tr_alpha1'] = 0.25
opts['tr_alpha2'] = 2
opts['tr_tol'] = 1e-8
opts['opt_check_slacks'] = True
# opts['verbose'] = False
opts['testproblem_obj'] = testproblem
opts['hess_lag_fun'] = testproblem.create_gn_hessian_python()
opts['n_slacks_start'] = 6
opts['n_slacks_end'] = 6
opts['feas_tol'] = 1e-8
# opts['gradient_correction'] = True
# opts['single_shooting_correction'] = True

# %%

feasible_solver = fslp.FSLP_Method()
x_sol, f_sol = feasible_solver.solve(problem_dict, init_dict, opts)

testproblem.plot([x_sol, x_opt_ipopt], ['FSLP', 'IPOPT'])

print("Optimal time FP-SQP: ", testproblem.get_optimal_time(x_sol))
print("Optimal time IPOPT: ", testproblem.get_optimal_time(x_opt_ipopt))