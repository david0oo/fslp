"""
This function contains all the necessary input given to the solver
"""
# Import standard libraries
from __future__ import annotations # <-still need this.
from typing import TYPE_CHECKING
import casadi as cs
# Import self-written libraries
if TYPE_CHECKING:
    from .logger import Logger
    from .options import Options


class NLPProblem:

    def __init__(self, nlp_dict: dict, opts: Options):
        
        # ---------------- Symbolic Expressions ----------------
        # primal variables
        self.x = nlp_dict['x']
        # parameter
        if 'p' in nlp_dict:
            self.p = nlp_dict['p']
        else:
            self.p = cs.MX.sym('p', 0)
        # objective
        self.f = nlp_dict['f']
        # constraints
        self.g = nlp_dict['g']

        self.number_variables = self.x.shape[0]
        self.number_constraints = self.g.shape[0]
        self.number_parameters = self.p.shape[0]

        # constraint multipliers
        self.lam_g = cs.MX.sym('lam_g', self.number_constraints)
        # bound constraint multipliers
        self.lam_x = cs.MX.sym('lam_x', self.number_variables)
        self.one = cs.MX.sym('one', 1)

        self.jacobian_g = cs.jacobian(self.g, self.x)
        self.gradient_f = cs.gradient(self.f, self.x)
        self.lagrangian = self.one*self.f + self.lam_g.T @ self.g +\
            self.lam_x.T @ self.x

        self.f_function = cs.Function('f_fun', [self.x, self.p], [self.f])

        self.gradient_f_function = cs.Function('grad_f_fun', [self.x, self.p], 
                                            [self.gradient_f])

        self.g_function = cs.Function('g_fun', [self.x, self.p], [self.g])

        self.jacobian_g_function = cs.Function('jac_g_fun', [self.x, self.p],
                                          [self.jacobian_g])

        self.gradient_lagrangian_function = cs.Function('grad_lag_fun',
                                                        [self.x, self.p, self.one, self.lam_g, self.lam_x],
                                                        [cs.gradient(self.lagrangian, self.x)])

        # ----------------------- still needs some refactoring here -----------

        if opts.use_sqp and 'hess_lag_fun' in opts:
            self.hessian_lagrangian_function = opts['hess_lag_fun']
        else:
            self.hessian_lagrangian_function = cs.Function('hess_lag_fun',
                                        [self.x, self.p, self.one, self.lam_g],
                                        [cs.hessian(
                                            self.lagrangian,
                                            self.x)[0]])

    def initialize(self, init_dict: dict):
        """
        Initialize the bounds, parameters, etc....
        """
        # ----------------- Bounds of problem ----------------------------
        if 'lbg' in init_dict:
            self.lbg = init_dict['lbg']
        else:
            self.lbg = -cs.inf*cs.DM.ones(self.number_constraints, 1)

        if 'ubg' in init_dict:
            self.ubg = init_dict['ubg']
        else:
            self.ubg = cs.inf*cs.DM.ones(self.number_constraints, 1)

        # Variable bounds
        if 'lbx' in init_dict:
            self.lbx = init_dict['lbx']
        else:
            self.lbx = -cs.inf*cs.DM.ones(self.number_variables, 1)

        if 'ubx' in init_dict:
            self.ubx = init_dict['ubx']
        else:
            self.ubx = cs.inf*cs.DM.ones(self.number_variables, 1)

    def eval_f(self, x: cs.DM, p: cs.DM, log: Logger):
        """
        Evaluates the objective function. And stores the statistics of it.

        Args:
            x (Casadi DM vector): the value of the states where f is evaluated

        Returns:
            Casadi DM scalar: the value of f at the given x.
        """
        log.increment_n_eval_f()
        return self.f_function(x, p)

    def eval_g(self, x: cs.DM, p: cs.DM, log: Logger):
        """
        Evaluates the constraint function. And stores the statistics of it.

        Args:
            x (Casadi DM vector): the value of the states where g is evaluated

        Returns:
            _type_: _description_
        """
        log.increment_n_eval_g()
        return self.g_function(x, p)

    def eval_gradient_f(self, x: cs.DM, p: cs.DM, log: Logger):
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
        return self.gradient_f_function(x, p)

    def eval_jacobian_g(self, x: cs.DM, p: cs.DM, log: Logger):
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
        return self.jacobian_g_function(x, p)

    def eval_gradient_lagrangian(self, 
                                   x: cs.DM,
                                   p: cs.DM,
                                   one: cs.DM,
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
        return self.gradient_lagrangian_function(x, p, one, lam_g, lam_x)

    def eval_hessian_lagrangian(self,
                                x: cs.DM,
                                p: cs.DM,
                                one: cs.DM,
                                lam_g: cs.DM,
                                log: Logger):
        """
        Evaluates the Hessian of Lagrangian. And stores the statistics 
        of it.
        """
        log.increment_n_eval_hessian_lagrangian()
        return self.hessian_lagrangian_function(x, p, one, lam_g)
