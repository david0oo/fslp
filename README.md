# fslp - feasible sequential linear programming: a prototypical implementation 
A prototypical implementation of the recently proposed Feaaible Sequential Linear Programming (FSLP) algorithm.

## installation on linux
In order to use the solver you need to install the following dependencies:
- The protoypical solver is written in `python3`. Therefore, get a recent `python3` version
- Install `casadi`. You can follow the instructions on the <a href="https://web.casadi.org/get/">casadi website</a>.

The easiest way is done by
```bash
pip install casadi
```
- Get `CPLEX` version 12.8 from the <a href="https://www.ibm.com/support/pages/downloading-ibm-ilog-cplex-optimization-studio-v1280">IBM website</a>. Version 12.8 is the latest release that is compatible with `casadi`. `CPLEX` needs to be interfaced with `casadi`. If you compile casadi from source, please check out the following <a href="https://github.com/casadi/casadi/issues/2440">link</a> for assistance to interface the solver

## usage of the solver
