"""
This file defines the parent class iterate
"""
# Import standard libraries
from __future__ import annotations # <-still need this.
from typing import TYPE_CHECKING
import casadi as cs
import numpy as np
# Import self-written libraries
if TYPE_CHECKING:
    from fslp.nlp_problem import NLPProblem
    from fslp.options import Options
    from fslp.logger import Logger
    from fslp.direction import Direction
# from fslp.trustRegion import TrustRegion


class Iterate:

    def __init__(self,
                 problem: NLPProblem):
        """
        Constructor.
        """
        self.number_variables = problem.number_variables
        self.number_constraints = problem.number_constraints
        self.number_parameters = problem.number_parameters

        self.x_k = cs.DM.zeros(self.number_variables, 1)
        self.lam_g_k = cs.DM.zeros(self.number_constraints, 1)
        self.lam_x_k = cs.DM.zeros(self.number_variables, 1)
        self.p = cs.DM.zeros(self.number_parameters, 1)
        # Initialize x, lam_g, lam_x, ....
        # self.__initialize(initialization_dict, problem, log)

    def initialize(self,
                     initialization_dict: dict,
                     problem: NLPProblem,
                     options: Options,
                     log:Logger):

        # Define iterative variables
        if 'x0' in initialization_dict:
            self.x_k = initialization_dict['x0']

        if 'lam_g0' in initialization_dict:
            self.lam_g_k = initialization_dict['lam_g0']

        if 'lam_x0' in initialization_dict:
            self.lam_x_k = initialization_dict['lam_x0']

        if 'p0' in initialization_dict:
            self.p = initialization_dict['p0']

        # Initialize the inner iterates as well
        self.x_inner_iterates = cs.DM.zeros(problem.number_variables)

        self.evaluate_quantities(problem, log, options)

    def evaluate_quantities(self,
                            nlp_problem: NLPProblem,
                            log: Logger,
                            parameter: Options):
        
        # Evaluate functions
        self.f_k = nlp_problem.eval_f(self.x_k, self.p, log)
        self.g_k = nlp_problem.eval_g(self.x_k, self.p, log) 
        self.gradient_f_k = nlp_problem.eval_gradient_f(self.x_k, self.p, log)
        self.jacobian_g_k = nlp_problem.eval_jacobian_g(self.x_k, self.p, log)
        self.gradient_lagrangian_k = nlp_problem.eval_gradient_lagrangian(self.x_k, self.p, cs.DM([1]), self.lam_g_k, self.lam_x_k, log)

        if parameter.use_sqp:
            self.hessian_lagrangian_k = nlp_problem.eval_hessian_lagrangian(self.x_k, self.p, cs.DM([1]), self.lam_g_k, log)

        # Calculate current infeasibility
        self.infeasibility = self.feasibility_measure(self.x_k, self.g_k, nlp_problem)

    def complementarity_condition(self, problem: NLPProblem):
        """
        Calculate inf-norm of the complementarity condition.
        """

        pre_compl_g = cs.fmin(cs.fabs(problem.lbg - self.g_k),
                              cs.fabs(self.g_k - problem.ubg))
        # is this sufficient?? because if there is no constraint we should not
        # care about it
        pre_compl_g[list(np.nonzero(pre_compl_g == cs.inf)[0])] = 0

        compl_g = cs.mmax(pre_compl_g * self.lam_g_k)

        pre_compl_x = cs.fmin(cs.fabs(problem.lbx - self.x_k),
                              cs.fabs(self.x_k - problem.ubx))

        # same here. Is this sufficient??
        pre_compl_x[list(np.nonzero(pre_compl_x == cs.inf)[0])] = 0

        compl_x = cs.mmax(pre_compl_x * self.lam_x_k)

        return cs.fmax(compl_g, compl_x)

    def feasibility_measure(self,
                            x: cs.DM,
                            g_x: cs.DM,
                            nlp_problem: NLPProblem):
        """
        The feasibility measure in the l-\\infty norm.

        Args:
            x (DM-array): value of state variable
            g_x (DM-array): value of constraints at state variable

        Returns:
            double: the feasibility in the l-\\infty norm
        """
        return float(cs.norm_inf(cs.vertcat(
                        cs.fmax(0, nlp_problem.lbg-g_x),
                        cs.fmax(0, g_x-nlp_problem.ubg),
                        cs.fmax(0, nlp_problem.lbx-x),
                        cs.fmax(0, x-nlp_problem.ubx))))
    
    def step_update(self, step_acceptable: bool, direction: Direction):
        """
        Args:
            direction (Direction): _description_
        """ 
        if step_acceptable:
            self.x_k = self.x_inner_iterates
            self.lam_x_k = direction.lam_d_inner_iterates
            self.lam_g_k = direction.lam_a_inner_iterates