"""
Implements an illustrative example showing the performance of FSLP on an NLP
that is fully determined and an NLP that is not fully determined.

In particular the problems are defined by:

min     x2
s.t.    x2 >= x1**2,
        x2 >= 0.1x1 + epsilon.

In the case of the undetermined system, we use epsilon=-0.06 and for the fully
determined case, we use epsilon=0.06.
"""
# %% Import statements
import casadi as cs
from fslp import fslp
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


def latexify():
    params = {#'backend': 'ps',
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


# %% Create and solve undetermined optimization problem
x = cs.MX.sym('x', 2)
f = x[1]
g = cs.vertcat(x[1] - x[0]**2, x[1] - 0.1*x[0] + 0.06)
lbg = cs.vertcat(0, 0)
ubg = cs.vertcat(cs.inf, cs.inf)
lbx = -cs.inf
ubx = cs.inf
# x0 = cs.vertcat(2, 10)
x0 = cs.vertcat(2, 5)

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
init_dict['tr_rad0'] = 5.5#1

opts = {}
opts['solver_type'] = 'SLP'
opts['subproblem_sol'] = 'cplex'
opts['subproblem_sol_opts'] = {'verbose': True,
                      'tol': 1e-9,
                      'qp_method': 2,
                      'warm_start': True,
                      'dep_check': 2,
                      'cplex': {'CPXPARAM_Simplex_Display': 0,
                                'CPXPARAM_ScreenOutput': 0}}
opts['max_iter'] = 1
opts['optim_tol'] = 1e-12


# opts = {}
# opts['solver_type'] = 'SQP'
# opts['subproblem_sol'] = 'qpoases'
# opts['subproblem_sol_opts'] = {'nWSR':10000, "sparse":True, 'hessian_type': 'semidef', 'printLevel': 'none'}
# opts['max_iter'] = 500
# opts['optim_tol'] = 1e-10

# Create FSLP solver
feasible_solver = fslp.FSLP_Method()
x_sol, f_sol = feasible_solver.solve(problem_dict, init_dict, opts)
m_k_undetermined = feasible_solver.list_mks

# %% Create and solve fully determined NLP
g = cs.vertcat(x[1] - x[0]**2, x[1] - 0.1*x[0]-0.06)
lbg = cs.vertcat(0, 0)
ubg = cs.vertcat(cs.inf, cs.inf)
problem_dict['g'] = g
init_dict['lbg'] = lbg
init_dict['ubg'] = ubg
x_sol, f_sol = feasible_solver.solve(problem_dict, init_dict, opts)
m_k_determined = feasible_solver.list_mks

# %% Plot Contraction of convergence criterion
plt.figure(figsize=(5, 3))
iters = np.arange(0, len(m_k_undetermined))
iter_feas = m_k_undetermined
plt.semilogy(iters,
             iter_feas,
             label=r'Not fully determined NLP, $\varepsilon =-0.06$',
             marker='.')
iters = np.arange(0, len(m_k_determined))
iter_feas = m_k_determined
plt.semilogy(iters,
             iter_feas,
             label=r'Fully Determined NLP, $\varepsilon =0.06$',
             linestyle='dashed',
             marker='^')
plt.xlabel('outer iteration number')
plt.xlim((0, len(m_k_undetermined)))
plt.ylabel('infinity norm of linear model $\Vert m_k \Vert_{\infty}$')
plt.grid(alpha=0.5)
plt.tight_layout()
plt.legend()
# plt.savefig(f"illustrative_example.pdf",
#             dpi=300,
#             bbox_inches='tight',
#             pad_inches=0.01)
plt.show()
