"""
This file defines the parent class iterate
"""
import casadi as cs
import numpy as np
from .input import Input
from .functionEvaluator import FunctionEvaluator
from .parameter import Parameter
from .logger import Logger
from .direction import Direction

class Iterate:

    def __init__(self,
                 input: Input,
                 parameter: Parameter,
                 log: Logger,
                 functionEvaluator: FunctionEvaluator):
        """
        Constructor.
        """
        # Store initial points
        self.x_k = input.x0
        self.lam_g_k = input.lam_g0
        self.lam_x_k = input.lam_x0
        self.p = input.p

        # Evaluate functions
        self.f_k = functionEvaluator.__eval_f(self.x_k, self.p, log)
        self.g_k = functionEvaluator.__eval_g(self.x_k, self.p, log)
        self.grad_f_k = functionEvaluator.__eval_gradient_f(self.x_k, self.p, log)
        self.jac_g_k = functionEvaluator.__eval_jacobian_g(self.x_k, self.p, log)

        if parameter.use_sqp:
            self.hess_lag_k = functionEvaluator.__eval_hessian_lagrangian(self.x_k, self.p, self.lam_g_k, log)


        # Calculate current infeasibility
        self.infeasibility = self.feasibility_measure(self.x_k, self.g_k)

        # Use this for scaling
        self.regularization_factor = 0

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

    def feasibility_measure(self, x: cs.DM, g_x: cs.DM, input: Input):
        """
        The feasibility measure in the l-\\infty norm.

        Args:
            x (DM-array): value of state variable
            g_x (DM-array): value of constraints at state variable

        Returns:
            double: the feasibility in the l-\\infty norm
        """
        return float(cs.norm_inf(cs.vertcat(
                        cs.fmax(0, input.lbg-g_x),
                        cs.fmax(0, g_x-input.ubg),
                        cs.fmax(0, input.lbx-x),
                        cs.fmax(0, x-input.ubx))))
    
    def update_primal_variables(self, direction: Direction):