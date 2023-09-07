"""
This file handles the output of the Feasible Sequential Linear/Quadratic
Programming solver
"""
# Import standard libraries
from __future__ import annotations # <-still need this.
from typing import TYPE_CHECKING
import casadi as cs
import numpy as np
# Import self-written libraries
if TYPE_CHECKING:
    from fslp.iterate import Iterate
    from fslp.direction import Direction
    from fslp.nlp_problem import NLPProblem
    from fslp.logger import Logger
    from fslp.trustRegion import TrustRegion


class Output:

    def __init__(self):
        pass

    def print_header(self):
        """
        This is the algorithm header
        """
        print("--------------------------------------------------------------")
        print("                        This is FSLP                          ")
        print("               © MECO Research Team, KU Leuven                ")
        print("--------------------------------------------------------------")

    def print_iteration_header(self):
        """ 
        Prints the iteration header.
        """
        print(("{iter:>6} | {m_k:^10} | {grad_lag:^10} | {feas:^10} | "
                   "{compl:^10} | {f_x:^10} | {lam_g:^13} | {lam_x:^13} | "
                   "{feas_it:^9} | {tr_rad:^13} | {tr_ratio:^10}").format(
                        iter='iter', m_k='m_k', grad_lag='grad_lag',
                        compl='compl', f_x='f(x)', lam_g='||lamg||_inf',
                        lam_x='||lamx||_inf', feas='infeas',
                        feas_it='feas iter', tr_rad='tr_rad',
                        tr_ratio='tr_ratio'))

    def print_output(self,
                     i: int,
                     iterate: Iterate,
                     direction: Direction,
                     problem: NLPProblem,
                     tr: TrustRegion,
                     log: Logger):
        """
        This function prints the iteration output to the console.

        Args:
            i (integer): the iteration index of the solve operator.
        """
        if i % 10:
            self.print_iteration_header()

        print(("{iter:>6} | {m_k:^10.4e} | {grad_lag:^10.4e} | {feas:^10.4e} | "
               "{compl:^10.4e} | {f_x:^10.4f} | {lam_g:^13.4e} | "
               "{lam_x:^13.4e} | {feas_it:^9} | {tr_rad:^13.5e} | "
               "{tr_ratio:^10.8f}").format(
                     iter=i, m_k=float(direction.m_k),
                     grad_lag=float(cs.norm_inf(iterate.gradient_lagrangian_k)),
                     compl=float(
                         iterate.complementarity_condition(problem)),
                     f_x=float(iterate.f_k),
                     lam_g=float(cs.norm_inf(iterate.lam_g_k)),
                     lam_x=float(cs.norm_inf(iterate.lam_x_k)),
                     feas=float(iterate.infeasibility),
                     feas_it=int(log.inner_iters[-1]),
                     tr_rad=float(log.tr_radii[-1]),
                     tr_ratio=float(tr.tr_ratio)))
        
    def print_feasibility_iterations_info(self,
                                          kappa: float,
                                          infeasibility: float,
                                          asymptotic_exactness: float):
        """
        Print the inner iteration debug output.
        """
        # print("Kappa: ", kappa,
        #       "Infeasibility", infeasibility,
        #       "Asymptotic Exactness: ", asymptotic_exactness)
        
        print(("Kappa:{kappa:>10.6f}"
              "    Infeasibility:{infeasibility:>15.5e}"
              "    Asymptotic Exactness: {asymptotic_exactness:10.6f}").format(
                  kappa=float(kappa),
                  infeasibility=float(infeasibility),
                  asymptotic_exactness=float(asymptotic_exactness)
              ))
        
