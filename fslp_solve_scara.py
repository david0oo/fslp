import casadi as cs
import numpy as np
from src.fslp import fslp
from timeit import default_timer as timer

from examples.scara_problems.P2P_timeoptimal_accelred import scara_problem as tp
# from P2P_approx_timeoptimal_accelred import scara_problem as tp
# from P2P_doubleinteg_cartesian import scara_problem as tp
# from P2P_singleinteg import single_integrator as tp

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def latexify():
    params = {'backend': 'ps',
            #'text.latex.preamble': r"\usepackage{amsmath}",
            'axes.labelsize': 10,
            'axes.titlesize': 10,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'text.usetex': True,
            'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)

latexify()

class MyCallback(cs.Callback):
  def __init__(self, name, nx, ng, np, opts={}):
    cs.Callback.__init__(self)

    self.times = []
    self.x_s = []
    self.nx = nx
    self.ng = ng
    self.np = np
    # Initialize internal objects
    self.construct(name, opts)

  def get_n_in(self): return cs.nlpsol_n_out()
  def get_n_out(self): return 1
  def get_name_in(self, i): return cs.nlpsol_out(i)
  def get_name_out(self, i): return "ret"

  def get_sparsity_in(self, i):
    n = cs.nlpsol_out(i)
    if n=='f':
        return cs.Sparsity.scalar()
    elif n in ('x', 'lam_x'):
        return cs.Sparsity.dense(self.nx)
    elif n in ('g', 'lam_g'):
        return cs.Sparsity.dense(self.ng)
    else:
        return cs.Sparsity(0,0)
        
  def eval(self, arg):
    # Create dictionary
    darg = {}
    for (i,s) in enumerate(cs.nlpsol_out()): darg[s] = arg[i]

    sol = darg['x']
    # self.x_sols.append(float(sol[0]))
    # self.y_sols.append(float(sol[1]))

    # if hasattr(self,'lines'):
    #   if "template" not in matplotlib.get_backend(): # Broken for template: https://github.com/matplotlib/matplotlib/issues/8516/
    #     self.lines[0].set_data(self.x_sols,self.y_sols)

    # else:
    #   self.lines = plot(self.x_sols,self.y_sols,'or-')

    # draw()
    # time.sleep(0.25)
    self.x_s.append(sol)
    self.times.append(timer())
    return [0]

def get_n_active_constraints(g_x, lbg, ubg):
    # lbx_active = x-lbx <= 0
    # ubx_active = ubx-x <= 0
    # nx_active = np.count_nonzero(np.logical_or(lbx_active, ubx_active))
    lbg_active = g_x-lbg <= 1e-8
    ubg_active = ubg-g_x <= 1e-8
    ng_active = np.count_nonzero(np.logical_or(np.array(lbg_active).squeeze(), np.array(ubg_active).squeeze()))
    n_active_constraints = ng_active
    return n_active_constraints


n_slacks_start = 4
n_slacks_end = 4

def feas_slacks(x_k):
    s0 = x_k[0:n_slacks_start]
    sf = x_k[-(1+n_slacks_end):-1]
    s = cs.vertcat(s0, sf)

    return cs.norm_inf(s)


testproblem = tp(N=25)
# (x, f, g, lbg, ubg, lbx, ubx, x0) = testproblem.create_problem(dev_lx_0=start_list_tuple, dev_lx_f=end_list_tuple)
(x, f, g, lbg, ubg, lbx, ubx, x0) = testproblem.create_problem(obstacleAvoidance=True)

nlp = {'x': x, 'f': f, 'g': g}
opts_ipopt = {'ipopt': {
    'fixed_variable_treatment': 'make_constraint', 'linear_solver': 'ma57'}}
mycallback = MyCallback('mycallback', x.shape[0], g.shape[0], 0)
opts_ipopt['iteration_callback'] = mycallback
opts_sqpmethod = {'convexify_strategy': 'regularize', 'max_iter': 200}
ipopt_solver = cs.nlpsol('ipopt_solver', 'ipopt', nlp, opts_ipopt)
sol = ipopt_solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
# # The optimal solution of the NLP..This solution should be found
x_opt_ipopt = sol['x']
lam_g_opt_ipopt = sol['lam_g']

start_time = mycallback.times[0]
ipopt_times = [i - start_time for i in mycallback.times]


ipopt_inf_pr = ipopt_solver.stats()['iterations']['inf_pr']
# ipopt_inf_pr.insert(0, 0.0)
ipopt_obj = ipopt_solver.stats()['iterations']['obj']

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
init_dict['tr_rad0'] = 0.125#1.0

# Time optimal crane unslacked state constraints
init_dict['tr_scale_mat0'], init_dict['tr_scale_mat_inv0'] = testproblem.create_scaling_matrices()

opts = {}
opts['lpsol'] = 'cplex'
opts['lpsol_opts'] = {'verbose': False, 'tol': 1e-9, 'qp_method': 2, 'warm_start': True,
                      'dep_check': 0, 'cplex': {'CPXPARAM_Simplex_Display': 0, 'CPXPARAM_ScreenOutput': 0}}
opts['max_iter'] = 100
opts['optim_tol'] = 1e-8
opts['max_inner_iter'] = 100
opts['tr_eta1'] = 0.25
opts['tr_eta2'] = 0.75
opts['tr_alpha1'] = 0.25
opts['tr_alpha2'] = 2
opts['tr_tol'] = 1e-8
opts['feas_tol'] = 1e-7
opts['opt_check_slacks'] = True
opts['n_slacks_start'] = 4
opts['n_slacks_end'] = 4
# opts['gradient_correction'] = True
# opts['verbose'] = False
opts['testproblem_obj'] = testproblem

feasible_solver = fslp.FSLP_Method()
x_sol, f_sol = feasible_solver.solve(problem_dict, init_dict, opts)
m_k = feasible_solver.list_mks
testproblem.plot_trajectory([testproblem.get_state_sol(x_opt_ipopt),
                             testproblem.get_state_sol(x_sol)],
                            ['ipopt', 'fslp'])

testproblem.plot_controls(x_sol)
testproblem.plot_states(x_sol)

ipopt_stats = ipopt_solver.stats()
t_wall_total = ipopt_stats['t_wall_total']
t_wall_nlp_hess_l = ipopt_stats['t_wall_nlp_hess_l']
print('IPOPT t_wall_total: ', t_wall_total, 't_wall_hess_l: ', t_wall_nlp_hess_l)

stats = feasible_solver.stats
t_wall = stats['t_wall']
t_wall_zero_slacks = stats['t_wall_zero_slacks']
print('FSLP t_wall_total: ', t_wall, 't_wall_zero_slacks: ', t_wall_zero_slacks)


# %% Check for number of active constraints
print('Number of variables: ', testproblem.n_vars)
print('Number of constraints: ', g.shape[0])
print('Number of active constraints: ', get_n_active_constraints(feasible_solver.g_fun(x_sol), lbg,ubg))
print("Number of inner iterations: ", stats['inner_iter'])

# %% Plot the termination criterion
plt.figure(figsize=(5,2.85))
iters = np.arange(0, len(m_k))
iter_feas = m_k
plt.semilogy(iters, np.array(iter_feas).squeeze(), label=r'Not fully determined NLP, $\varepsilon =-0.06$', marker='.')
plt.xlabel('outer iteration number')
plt.xlim((0,len(m_k)))
plt.ylabel('infinity norm of linear model $\Vert m_k \Vert_{\infty}$')
plt.grid(alpha=0.5)
plt.tight_layout()
# plt.savefig(f"illustrative_example.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.show()

# %%
plt.figure(figsize=(5,5))
plt.subplot(211)
iters = np.arange(0, len(m_k))
iter_feas = m_k
plt.semilogy(np.array(feasible_solver.list_times).squeeze(), np.array(feasible_solver.list_feas).squeeze(), label=r'FSLP', marker='.')
plt.semilogy(np.array(ipopt_times).squeeze(), np.array(ipopt_inf_pr).squeeze(), label=r'IPOPT', marker='.')
# plt.xlabel('time [s]')
plt.xlim((0,ipopt_times[-1]))
plt.ylabel('inf-norm of infeasibility')
plt.grid(alpha=0.5)
# plt.tight_layout()
# plt.savefig(f"infeasibilities_per_time.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
# plt.show()

# %%
slack_feas_fslp = [feas_slacks(x_k) for x_k in feasible_solver.list_iter]
slack_feas_ipopt = [feas_slacks(x_k) for x_k in mycallback.x_s]


# plt.figure(figsize=(5,2.85))
plt.subplot(212)
iters = np.arange(0, len(m_k))
iter_feas = m_k
plt.semilogy(np.array(feasible_solver.list_times).squeeze(), np.array(slack_feas_fslp).squeeze(), label=r'FSLP', marker='.')
plt.semilogy(np.array(ipopt_times).squeeze(), np.array(slack_feas_ipopt).squeeze(), label=r'IPOPT', marker='.')
plt.xlabel('time [s]')
plt.xlim((0,ipopt_times[-1]))
plt.ylabel('inf-norm of slacks')
plt.legend()
plt.grid(alpha=0.5)
plt.tight_layout()
# plt.savefig(f"slack_infeasibilities_per_time.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.savefig(f"infeasibilities.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.show()

#%% 

combined_feas_ipopt = []
for i in range(len(slack_feas_ipopt)):
    combined_feas_ipopt.append(cs.fmax(slack_feas_ipopt[i], ipopt_inf_pr[i]))
combined_feas_fslp = []
for i in range(len(slack_feas_fslp)):
    combined_feas_fslp.append(cs.fmax(slack_feas_fslp[i], feasible_solver.list_feas[i]))


plt.figure(figsize=(5,2.85))
plt.semilogy(np.array(feasible_solver.list_times).squeeze(), np.array(combined_feas_fslp).squeeze(), label=r'FSLP', marker='.')
plt.semilogy(np.array(ipopt_times).squeeze(), np.array(combined_feas_ipopt).squeeze(), label=r'IPOPT', marker='.')
plt.xlabel('time [s]')
plt.xlim((0,ipopt_times[-1]))
plt.ylabel('combined inf-norm')
plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig(f"combined_infeasibility.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.show()