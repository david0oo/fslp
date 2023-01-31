"""
This function contains all the necessary input given to the solver
"""
import casadi as cs

class Input:

    def __init__(self, nlp_dict: dict, initialization_dict: dict) -> None:
        
        self.x = nlp_dict['x']
        self.nx = self.x.shape[0]

        # objective
        self.f = nlp_dict['f']

        # constraints
        self.g = nlp_dict['g']
        self.ng = self.g.shape[0]

        if 'lbg' in initialization_dict:
            self.lbg = initialization_dict['lbg']
        else:
            self.lbg = -cs.inf*cs.DM.ones(self.ng, 1)

        if 'ubg' in initialization_dict:
            self.ubg = initialization_dict['ubg']
        else:
            self.ubg = cs.inf*cs.DM.ones(self.ng, 1)

        # Variable bounds
        if 'lbx' in initialization_dict:
            self.lbx = initialization_dict['lbx']
        else:
            self.lbx = -cs.inf*cs.DM.ones(self.nx, 1)

        if 'ubx' in initialization_dict:
            self.ubx = initialization_dict['ubx']
        else:
            self.ubx = cs.inf*cs.DM.ones(self.nx, 1)

        # Define iterative variables
        if 'x0' in initialization_dict:
            self.x0 = initialization_dict['x0']
        else:
            self.x0 = cs.DM.zeros(self.nx, 1)

        if 'lam_g0' in initialization_dict:
            self.lam_g0 = initialization_dict['lam_g0']
        else:
            self.lam_g0 = cs.DM.zeros(self.ng, 1)

        if 'lam_x0' in initialization_dict:
            self.lam_x0 = initialization_dict['lam_x0']
        else:
            self.lam_x0 = cs.DM.zeros(self.nx, 1)

        if 'tr_rad0' in initialization_dict:
            self.tr_rad0 = initialization_dict['tr_rad0']
        else:
            self.tr_rad0 = 1.0

        if 'tr_scale_mat0' in initialization_dict:
            self.tr_scale_mat0 = initialization_dict['tr_scale_mat0']
        else:
            self.tr_scale_mat0 = cs.DM.eye(self.nx)

        if 'tr_scale_mat_inv0' in initialization_dict:
            self.tr_scale_mat_inv0 = initialization_dict['tr_scale_mat_inv0']
        else:
            self.tr_scale_mat_inv0 = cs.inv(self.tr_scale_mat0)

        self.lam_g = cs.MX.sym('lam_g', self.ng)
        self.lam_x = cs.MX.sym('lam_x', self.nx)

        self.jac_g = cs.jacobian(self.g, self.x)
        self.grad_f = cs.gradient(self.f, self.x)
        self.lagrangian = self.f + self.lam_g.T @ self.g +\
            self.lam_x.T @ self.x

        self.f_fun = cs.Function('f_fun', [self.x], [self.f])

        self.grad_f_fun = cs.Function('grad_f_fun', [self.x], [self.grad_f])

        self.g_fun = cs.Function('g_fun', [self.x], [self.g])

        self.jac_g_fun = cs.Function('jac_g_fun', [self.x], [self.jac_g])

        self.grad_lag_fun = cs.Function('grad_lag_fun',
                                        [self.x, self.lam_g, self.lam_x],
                                        [cs.gradient(self.lagrangian,
                                                     self.x)])
                                                    
        if self.use_sqp and bool(opts) and 'hess_lag_fun' in opts:
            self.hess_lag_fun = opts['hess_lag_fun']
        else:
            self.hess_lag_fun = cs.Function('hess_lag_fun',
                                        [self.x, self.lam_g, self.lam_x],
                                        [cs.hessian(
                                            self.lagrangian,
                                            self.x)[0]])