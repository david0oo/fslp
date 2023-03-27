"""
This file is used as a function evaluator. All the functions given as input
are evaluated in this class
"""
import casadi as cs
from .logger import Logger
from .input import Input

class FunctionEvaluator:

    def __init__(self, input: Input) -> None:
        """ 
        Constructor of the class. All necessary functions are created here.
        """

        self.jac_g = cs.jacobian(input.g, input.x)
        self.grad_f = cs.gradient(input.f, input.x)
        self.lagrangian = input.f + input.lam_g.T @ input.g +\
            input.lam_x.T @ input.x

        self.f_casadi_function = cs.Function('f_fun', [input.x, input.p], [input.f])

        self.grad_f_casadi_function = cs.Function('grad_f_fun', [input.x, input.p], [self.grad_f])

        self.g_casadi_function = cs.Function('g_fun', [input.x, input.x], [input.g])

        self.jac_g_casadi_function = cs.Function('jac_g_fun', [input.x, input.p], [self.jac_g])

        self.grad_lag_casadi_function = cs.Function('grad_lag_fun',
                                        [input.x, input.p, input.lam_g, input.lam_x],
                                        [cs.gradient(self.lagrangian,
                                                     input.x)])

        # ----------------------- still needs some refactoring here -----------

        if input.use_sqp and 'hess_lag_fun' in opts:
            self.hess_lag_fun = opts['hess_lag_fun']
        else:
            one = cs.MX.sym('one', 1)
            self.hess_lag_casadi_function = cs.Function('hess_lag_fun',
                                        [self.x, self.p, one, self.lam_g],
                                        [cs.hessian(
                                            self.lagrangian,
                                            input.x)[0]])


    def __eval_f(self, x: cs.DM, p: cs.DM, log: Logger):
        """
        Evaluates the objective function. And stores the statistics of it.

        Args:
            x (Casadi DM vector): the value of the states where f is evaluated

        Returns:
            Casadi DM scalar: the value of f at the given x.
        """
        log.increment_n_eval_f()
        return self.f_casadi_function(x, p)

    def __eval_g(self, x: cs.DM, p: cs.DM, log: Logger):
        """
        Evaluates the constraint function. And stores the statistics of it.

        Args:
            x (Casadi DM vector): the value of the states where g is evaluated

        Returns:
            _type_: _description_
        """
        log.increment_n_eval_g()
        return self.g_casadi_function(x, p)

    def __eval_gradient_f(self, x: cs.DM, p: cs.DM, log: Logger):
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
        return self.grad_f_casadi_function(x, p)

    def __eval_jacobian_g(self, x:cs.DM, p: cs.DM, log:Logger):
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
        return self.jac_g_casadi_function(x, p)

    def __eval_gradient_lagrangian(self, 
                                   x: cs.DM, 
                                   p: cs.DM, 
                                   lam_g: cs.DM, 
                                   lam_x: cs.DM, 
                                   log:Logger):
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
        return self.grad_lag_casadi_function(x, p, lam_g, lam_x)

    def __eval_hessian_lagrangian(self, x:cs.DM, p: cs.DM, lam_g:cs.DM, log:Logger):
        """
        Evaluates the Hessian of Lagrangian. And stores the statistics 
        of it.
        """
        log.increment_n_eval_hessian_lagrangian()
        return self.hess_lag_fun(x, p, 1, lam_g)
