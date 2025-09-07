# fslp - feasible sequential linear programming: a prototypical implementation 
A prototypical implementation of the recently proposed Feasible Sequential Linear Programming (FSLP) algorithm.

## Installation on Linux
In order to use the solver you need to install the following dependencies:
- The protoypical solver is written in `python3`. Therefore, get a recent `python3` version
- Install `casadi`. You can follow the instructions on the <a href="https://web.casadi.org/get/">casadi website</a>.

The easiest way is done by
```bash
pip install casadi
```
- Get `CPLEX` version 12.8 from the <a href="https://www.ibm.com/support/pages/downloading-ibm-ilog-cplex-optimization-studio-v1280">IBM website</a>. Version 12.8 is the latest release that is compatible with `casadi`. `CPLEX` needs to be interfaced with `casadi`. If you compile casadi from source, please check out the following <a href="https://github.com/casadi/casadi/issues/2440">link</a> for assistance to interface the solver
- The HiGHs solver is available through the CasADi develop branch

## Usage of the Solver
The optimization problem should be modelled in the following form:

```math
\begin{split}
\min_{x \in R^n} \quad      &f\\
\mathrm{s.t.}  \quad     &lbg <= g <= ubg,\\
            &lbx <= x <= ubx
\end{split}
```
where x is a symbolic casadi vector and f, g are symbolic casadi expressions. The bounds lbg, ubg, lbx, ubx are numeric casadi vectors. 

**IMPORTANT: The solver needs to be initialized at a feasible point, such that it works correctly!**

### Example Problem
We want to solve the following NLP:
```math
\begin{split}
\min_{x \in R^2}\quad       &x_2\\
\mathrm{s.t.}\quad         &x_2 >= x_1^2\\
                     &x_2 >= 0.1x_1 + 0.06
\end{split}
```
the algorithm is initialized at the point <img src="https://render.githubusercontent.com/render/math?math=x_0= (2,\,10)^{\top}">

FSLP can be used as follows:
```python
# import statements
import casadi as cs
from fslp import fslp

# Create symbolic expression and bound vectors
x = cs.MX.sym('x', 2)
f = x[1]
g = cs.vertcat(x[1] - x[0]**2, x[1] - 0.1*x[0] - 0.06)
lbg = cs.vertcat(0, 0)
ubg = cs.vertcat(cs.inf, cs.inf)
lbx = -cs.inf
ubx = cs.inf
x0 = cs.vertcat(2, 10)

# Create problem dict
problem_dict = {}
problem_dict['x'] = x
problem_dict['f'] = f
problem_dict['g'] = g

# Create initialization dictionary
init_dict = {}
init_dict['lbx'] = lbx
init_dict['ubx'] = ubx
init_dict['lbg'] = lbg
init_dict['ubg'] = ubg
init_dict['x0'] = x0

# Create options dictionary
opts = {}
opts['lpsol'] = 'cplex'
opts['lpsol_opts'] = {'verbose': False,
                      'tol': 1e-9,
                      'qp_method': 2,
                      'warm_start': True,
                      'dep_check': 2,
                      'cplex': {'CPXPARAM_Simplex_Display': 0,
                                'CPXPARAM_ScreenOutput': 0}}
opts['max_iter'] = 45
opts['optim_tol'] = 1e-12

# Create FSLP solver and solve problem
feasible_solver = fslp.FSLP_Method()
x_sol, f_sol = feasible_solver.solve(problem_dict, init_dict, opts)
```
### OCP example
For the implementation of an OCP with casadi's Opti stack, please have a look into the following [example](https://github.com/david0oo/fslp/blob/main/examples/P2P_timeoptimal_crane_problem.py). 

The feasible initialization was achieved by introducing slack variables on the initial and terminal constraint.

## Literature
The algorithm is described in:

[A Feasible Sequential Linear Programming Algorithm with Application to Time-Optimal Path Planning Problems](https://arxiv.org/abs/2205.00754)
David Kiessling, Andrea Zanelli, Armin Nurkanović, Joris Gillis, Moritz Diehl, Melanie Zeilinger, Goele Pipeleers, Jan Swevers
published in the Proceedings of the 61st IEEE Conference on Decision and Control

## Citing
```
@InProceedings{Kiessling2022,
  Title                    = {A Feasible Sequential Linear Programming Algorithm with Application to Time-Optimal Path Planning Problems},
  Author                   = {Kiessling, D. and Zanelli, A. and Nurkanovi\'c, A. and Gillis, J. and Diehl, M. and Zeilinger, M. and Pipeleers, G. and Swevers, J.},
  Booktitle                = {Proceedings of 61st IEEE Conference on Decision and Control},
  Year                     = {2022},
  Address                  = {Cancun, Mexico},
  Month                    = {December},
}

@InProceedings{Kiessling2023,
  Title                    = {Anderson Accelerated Feasible Sequential Linear Programming},
  Author                   = {Kiessling, D. and Pas, P. and Astudillo, A. and Patrinos, P. and Swevers, J.},
  Booktitle                = {Proceedings of the IFAC World Congress},
  Year                     = {2023},
  Address                  = {Yokohama, Japan},
}
```
## Contact
For questions, remarks, bugs please send an email to: [david.kiessling@kuleuven.be](david.kiessling@kuleuven.be)


