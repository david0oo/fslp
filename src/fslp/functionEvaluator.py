"""
This file is used as a function evaluator. All the functions given as input
are evaluated in this class
"""
import casadi as cs
from .logger import Logger
from .input import Input

class FunctionEvaluator:

    def __init__(self) -> None:
        pass

    def __eval_f(self, x:cs.DM, log:Logger):
        """
        Evaluates the objective function. And stores the statistics of it.

        Args:
            x (Casadi DM vector): the value of the states where f is evaluated

        Returns:
            Casadi DM scalar: the value of f at the given x.
        """
        log.increment_n_eval_f()
        return self.f_fun(x)

    def __eval_g(self, x:cs.DM, log:Logger):
        """
        Evaluates the constraint function. And stores the statistics of it.

        Args:
            x (Casadi DM vector): the value of the states where g is evaluated

        Returns:
            _type_: _description_
        """
        log.increment_n_eval_g()
        return self.g_fun(x)

    def __eval_gradient_f(self, x:cs.DM, log:Logger):
        """
        Evaluates the objective gradient function. And stores the statistics 
        of it.
        
        Args:
            x (Casadi DM vector): the value of the states where gradient of f 
            is evaluated

        Returns:
           Casadi DM vector: the value of g at the given x.
        """
        log.increment_n_eval_gradient_f()
        return self.grad_f_fun(x)

    def __eval_jacobian_g(self, x:cs.DM, log:Logger):
        """
        Evaluates the constraint jacobian function. And stores the
        statistics of it.

        Args:
            x (Casadi DM vector): the value of the states where Jacobian of g 
            is evaluated

        Returns:
            Casadi DM vector: the value of g at the given x.
        """
        log.increment_n_eval_jacobian_g()
        return self.jac_g_fun(x)

    def __eval_gradient_lagrangian(self, x:cs.DM, lam_g:cs.DM, lam_x:cs.DM, log:Logger):
        """
        Evaluates the gradient of the Lagrangian at x, lam_g, and lam_x.
        
        Args:
            x (Casadi DM vector): the value of the states where Jacobian of g 
            lam_g (Casadi DM vector): value of multipliers for constraints g
            lam_x (Casadi DM vector): value of multipliers for states x

        Returns:
            Casadi DM vector: the value of gradient of Lagrangian
        """
        log.increment_n_eval_gradient_lagrangian()
        return self.grad_lag_fun(x, lam_g, lam_x)

    def __eval_hessian_lagrangian(self, x:cs.DM, lam_g:cs.DM, lam_x:cs.DM, log:Logger):
        """
        Evaluates the Hessian of Lagrangian. And stores the statistics 
        of it.
        """
        log.increment_n_eval_hessian_lagrangian()
        return self.hess_lag_fun(x, lam_g, lam_x)
