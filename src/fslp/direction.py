""" 
Calculates the search direction of an iterate.
"""

class Direction:

    def __init__(self) -> None:
        pass

    def __prepare_subproblem_matrices(self):
        """
        Prepares the objective vector g and the constraint matrix A for the LP.
        I.e., min_{x}   g_k.T x
              s.t.  lba <= A_k*x <= uba,
                    lbx <= x <= ubx.
        """
        self.A_k = self.val_jac_g_k
        self.g_k = self.val_grad_f_k
        if self.use_sqp:
            if self.regularize:
                self.H_k = self.hess_lag_k
                self.regularization_factor = 0
                while min(np.linalg.eig(self.H_k)[0]) < self.regularization_factor_min and self.regularization_factor <= self.regularization_factor_max:
                    if self.regularization_factor == 0:
                        self.regularization_factor = self.regularization_factor_min
                    else:
                        self.regularization_factor *= self.regularization_factor_increase
                    self.H_k += self.regularization_factor*cs.DM.eye(self.nx)
                print('factor', self.regularization_factor)
            else:
                self.H_k = self.hess_lag_k

    def __prepare_subproblem_bounds_constraints(self):
        """
        Prepare the bounds for the constraints in the LP.
        I.e., min_{x}   g_k.T x
              s.t.  lba <= A_k*x <= uba,
                    lbx <= x <= ubx.
        Linearizing the constraints gives a constant part that needs to be
        added to the bounds.
        """
        self.lba_k = cs.vertcat(self.lbg - self.val_g_k)
        self.uba_k = cs.vertcat(self.ubg - self.val_g_k)

    def __prepare_subproblem_bounds_variables(self):
        """
        Prepare the bounds for the variables in the LP.
        I.e., min_{x}   g_k.T x
              s.t.  lba <= A_k*x <= uba,
                    lbx <= x <= ubx.
        The constant part of the linearization needs to be added in the bounds 
        and the trust-region needs to be taken into account as well.
        """
        lbp = cs.fmax(-self.tr_rad_k*self.tr_scale_mat_inv_k @
                      cs.DM.ones(self.nx, 1), self.lbx - self.x_k)
        ubp = cs.fmin(self.tr_rad_k*self.tr_scale_mat_inv_k @
                      cs.DM.ones(self.nx, 1), self.ubx - self.x_k)

        self.lb_var_k = lbp
        self.ub_var_k = ubp

    def eval_m_k(self, p:cs.DM):
        """
        In case of SQP:
        Evaluates the quadratic model of the objective function, i.e.,
        m_k(x_k; p) = grad_f_k.T @ p + p.T @ H_k @ p
        H_k denotes the Hessian Approximation

        In case of SLP:
        Evaluates the linear model of the objective function, i.e.,
        m_k(x_k; p) = grad_f_k.T @ p. This model is used in the termination
        criterion.

        Args:
            p (Casadi DM vector): the search direction where the linear model
                                  should be evaluated

        Returns:
            double: the value of the linear model at the given direction p.
        """
        if self.use_sqp:
            return self.val_grad_f_k.T @ p + 0.5 * p.T @ self.H_k @ p
        else:
            return self.val_grad_f_k.T @ p