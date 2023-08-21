"""
This function contains all the necessary input given to the solver
"""
import casadi as cs
from .logger import Logger


class NLPProblem:

    def __init__(self, nlp_dict: dict, opts: dict):
        
        # ---------------- Symbolic Expressions ----------------
        # primal variables
        self.x = nlp_dict['x']
        # parameter
        self.p = nlp_dict['p']
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

        self.jacobian_g = cs.jacobian(self.g, self.x)
        self.gradient_f = cs.gradient(self.f, self.x)
        self.lagrangian = self.f + self.lam_g.T @ self.g +\
            self.lam_x.T @ self.x

        self.f_function = cs.Function('f_fun', [self.x, self.p], [self.f])

        self.grad_f_function = cs.Function('grad_f_fun', [self.x, self.p], 
                                            [self.gradient_f])

        self.g_function = cs.Function('g_fun', [self.x, self.x], [self.g])

        self.jac_g_function = cs.Function('jac_g_fun', [self.x, self.p], 
                                          [self.jacobian_g])

        self.grad_lag_function = cs.Function('grad_lag_fun',
                                        [self.x, self.p, self.lam_g, self.lam_x],
                                        [cs.gradient(self.lagrangian,
                                                     self.x)])

        # ----------------------- still needs some refactoring here -----------

        if opts.use_sqp and 'hess_lag_fun' in opts:
            self.hess_lag_fun = opts['hess_lag_fun']
        else:
            one = cs.MX.sym('one', 1)
            self.hess_lag_casadi_function = cs.Function('hess_lag_fun',
                                        [self.x, self.p, one, self.lam_g],
                                        [cs.hessian(
                                            self.lagrangian,
                                            input.x)[0]])


        # ----------------- Bounds of problem ----------------------------
        if 'lbg' in nlp_dict:
            self.lbg = nlp_dict['lbg']
        else:
            self.lbg = -cs.inf*cs.DM.ones(self.number_constraints, 1)

        if 'ubg' in nlp_dict:
            self.ubg = nlp_dict['ubg']
        else:
            self.ubg = cs.inf*cs.DM.ones(self.number_constraints, 1)

        # Variable bounds
        if 'lbx' in nlp_dict:
            self.lbx = nlp_dict['lbx']
        else:
            self.lbx = -cs.inf*cs.DM.ones(self.number_variables, 1)

        if 'ubx' in nlp_dict:
            self.ubx = nlp_dict['ubx']
        else:
            self.ubx = cs.inf*cs.DM.ones(self.number_variables, 1)

    def __eval_f(self, x: cs.DM, p: cs.DM, log: Logger):
        """
        Evaluates the objective function. And stores the statistics of it.

        Args:
            x (Casadi DM vector): the value of the states where f is evaluated

        Returns:
            Casadi DM scalar: the value of f at the given x.
        """
        log.increment_n_eval_f()
        return self.f_function(x, p)

    def __eval_g(self, x: cs.DM, p: cs.DM, log: Logger):
        """
        Evaluates the constraint function. And stores the statistics of it.

        Args:
            x (Casadi DM vector): the value of the states where g is evaluated

        Returns:
            _type_: _description_
        """
        log.increment_n_eval_g()
        return self.g_function(x, p)

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
        return self.grad_f_function(x, p)

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
        return self.jac_g_function(x, p)

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
        return self.grad_lag_function(x, p, lam_g, lam_x)

    def __eval_hessian_lagrangian(self, x:cs.DM, p: cs.DM, lam_g:cs.DM, log:Logger):
        """
        Evaluates the Hessian of Lagrangian. And stores the statistics 
        of it.
        """
        log.increment_n_eval_hessian_lagrangian()
        return self.hess_lag_fun(x, p, 1, lam_g)
