"""
This file defines the parent class iterate
"""
import casadi as cs
import numpy as np
from .nlp_problem import NLPProblem
from .options import Options
from .logger import Logger
from .direction import Direction

class Iterate:

    def __init__(self,
                 input: NLPProblem,
                 parameter: Options,
                 log: Logger):
        """
        Constructor.
        """
        # Store initial points
        # self.x_k = input.x0
        # self.lam_g_k = input.lam_g0
        # self.lam_x_k = input.lam_x0
        # self.p = input.p

        self.number_variables = input.number_variables
        self.number_constraints = input.number_constraints
        self.number_parameters = input.number_parameters

        # Use this for scaling
        self.regularization_factor = 0

    def __initialize(self, initialization_dict: dict, nlp_problem:NLPProblem, log:Logger):

        # Define iterative variables
        if 'x0' in initialization_dict:
            self.x_k = initialization_dict['x0']
        else:
            self.x_k = cs.DM.zeros(self.number_variables, 1)

        if 'lam_g0' in initialization_dict:
            self.lam_g_k = initialization_dict['lam_g0']
        else:
            self.lam_g_k = cs.DM.zeros(self.number_constraints, 1)

        if 'lam_x0' in initialization_dict:
            self.lam_x_k = initialization_dict['lam_x0']
        else:
            self.lam_x_k = cs.DM.zeros(self.number_variables, 1)

        if 'p' in initialization_dict:
            self.p = initialization_dict['p']
        else:
            self.p = cs.DM(self.number_parameters, 1)


    def evaluate_quantities(self, nlp_problem: NLPProblem, log: Logger, parameter: Options)
        
        # Evaluate functions
        self.f_k = nlp_problem.__eval_f(self.x_k, self.p, log)
        self.g_k = nlp_problem.__eval_g(self.x_k, self.p, log)
        self.grad_f_k = nlp_problem.__eval_gradient_f(self.x_k, self.p, log)
        self.jac_g_k = nlp_problem.__eval_jacobian_g(self.x_k, self.p, log)

        if parameter.use_sqp:
            self.hess_lag_k = nlp_problem.__eval_hessian_lagrangian(self.x_k, self.p, self.lam_g_k, log)

        # Calculate current infeasibility
        self.infeasibility = self.feasibility_measure(self.x_k, self.g_k)


    def __complementarity_condition(self):
        """
        Calculate inf-norm of the complementarity condition.
        """

        pre_compl_g = cs.fmin(cs.fabs(self.lbg - self.val_g_k),
                              cs.fabs(self.val_g_k - self.ubg))
        # is this sufficient?? because if there is no constraint we should not
        # care about it
        pre_compl_g[list(np.nonzero(pre_compl_g == cs.inf)[0])] = 0

        compl_g = cs.mmax(pre_compl_g * self.lam_g_k)

        pre_compl_x = cs.fmin(cs.fabs(self.lbx - self.x_k),
                              cs.fabs(self.x_k - self.ubx))

        # same here. Is this sufficient??
        pre_compl_x[list(np.nonzero(pre_compl_x == cs.inf)[0])] = 0

        compl_x = cs.mmax(pre_compl_x * self.lam_x_k)

        return cs.fmax(compl_g, compl_x)

    def feasibility_measure(self, x: cs.DM, g_x: cs.DM, nlp_problem: NLPProblem):
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
    
    # def __eval_grad_jac(self, step_accepted: bool=False):
    #     """ 
    #     Evaluate functions, gradient, jacobian at current iterate x_k.

    #     Args:
    #         step_accepted (bool, optional): Denotes if previous step was
    #         accepted. In an accepted step the gradient of the constraints do
    #         not need to be re-evaluated. Defaults to False.
    #     """
    #     self.val_f_k = self.__eval_f(self.x_k)
    #     if step_accepted:
    #         self.val_g_k = self.g_tmp
    #     else:
    #         self.val_g_k = self.__eval_g(self.x_k)
    #     self.val_grad_f_k = self.__eval_grad_f(self.x_k)
    #     self.val_jac_g_k = self.__eval_jac_g(self.x_k)
    #     if self.use_sqp:
    #         self.hess_lag_k = self.__eval_hess_l(self.x_k,
    #                                              self.lam_g_k,
    #                                              self.lam_x_k)


    def update_primal_variables(self, direction: Direction):
        """
        Args:
            direction (Direction): _description_
        """ 