"""
This file handles the output of the Feasible Sequential Linear/Quadratic
Programming solver
"""
# Import standard libraries
import casadi as cs
import numpy as np
# Import self-written libraries
from .iterate import Iterate
from .direction import Direction
from .nlp_problem import NLPProblem
from .logger import Logger
from .trustRegion import TrustRegion


class Output:

    def __init__(self):
        pass

    def print_header(self):
        """
        This is the algorithm header
        """
        print("-------------------------------------")
        print("           This is FSLP              ")
        print("-------------------------------------")

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
                     iter=i, m_k=np.array(direction.m_k).squeeze(),
                     grad_lag=np.array(cs.norm_inf(
                         problem.eval_gradient_lagrangian(
                             iterate.x_k, iterate.lam_g_k, iterate.lam_x_k, log))).squeeze(),
                     compl=np.array(
                         iterate.complementarity_condition(problem)).squeeze(),
                     f_x=np.array(iterate.f_k).squeeze(),
                     lam_g=np.array(cs.norm_inf(iterate.lam_g_k)).squeeze(),
                     lam_x=np.array(cs.norm_inf(iterate.lam_x_k)).squeeze(),
                     feas=iterate.infeasibility,
                     feas_it=log.inner_iters,
                     tr_rad=np.array(log.tr_radii[-1]).squeeze(),
                     tr_ratio=np.array(tr.rho_k).squeeze()))
        
