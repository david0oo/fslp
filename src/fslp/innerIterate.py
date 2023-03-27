""" 
Describes an outer iterate of the algorithm.
"""
import casadi as cs
import numpy as np
from .input import Input
from .iterate import Iterate

class InnerIterate(Iterate):

    def __init__(self, input: Input) -> None:
        self.x_k = input.x0
        self.lam_g_k = input.lam_g0
        self.lam_x_k = input.lam_x0

        self.infeasibility = self.feasibility_measure(self.x_k, self.g_k)

    def __eval_grad_jac(self, step_accepted: bool=False):
        """ 
        Evaluate functions, gradient, jacobian at current iterate x_k.

        Args:
            step_accepted (bool, optional): Denotes if previous step was
            accepted. In an accepted step the gradient of the constraints do
            not need to be re-evaluated. Defaults to False.
        """
        self.val_f_k = self.__eval_f(self.x_k)
        if step_accepted:
            self.val_g_k = self.g_tmp
        else:
            self.val_g_k = self.__eval_g(self.x_k)
        self.val_grad_f_k = self.__eval_grad_f(self.x_k)
        self.val_jac_g_k = self.__eval_jac_g(self.x_k)
        if self.use_sqp:
            self.hess_lag_k = self.__eval_hess_l(self.x_k,
                                                 self.lam_g_k,
                                                 self.lam_x_k)