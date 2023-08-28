# Import standard libraries
import casadi as cs
import numpy as np
# Import self-written libraries
from .nlp_problem import NLPProblem
from .iterate import Iterate
from .options import Options
from .trustRegion import TrustRegion


class Direction:

    def __init__(self,
                 problem: NLPProblem,
                 opts: Options):
        
        # Initialize the direction
        self.d_k = cs.DM.zeros(problem.number_variables)
        self.lam_d_k = cs.DM.zeros(problem.number_variables)
        self.lam_a_k = cs.DM.zeros(problem.number_constraints)

        self.d_inner_iterates = cs.DM.zeros(problem.number_variables)
        self.lam_d_inner_iterates = cs.DM.zeros(problem.number_variables)
        self.lam_a_inner_iterates = cs.DM.zeros(problem.number_constraints)
        self.use_sqp = opts.use_sqp

    def eval_m_k(self,
                 iterate: Iterate):
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
            self.m_k = iterate.gradient_f_k.T @ self.d_k + 0.5 * self.d_k.T @ self.H_k @ self.d_k
        else:
            self.m_k = iterate.gradient_f_k.T @ self.d_k
        
    def prepare_subproblem_matrices(self,
                                    iterate: Iterate,
                                    opts: Options):
        """
        Prepares the objective vector g and the constraint matrix A for the LP.
        I.e., min_{x}   g_k.T x
              s.t.  lba <= A_k*x <= uba,
                    lbx <= x <= ubx.
        """
        self.A_k = iterate.jacobian_g_k
        self.objective_vector_k = iterate.gradient_f_k
        if self.use_sqp:
            if opts.regularize:
                self.H_k = iterate.hessian_lagrangian_k
                self.regularization_factor = 0.0
                while min(np.linalg.eig(self.H_k)[0]) < opts.regularization_factor_min and self.regularization_factor <= opts.regularization_factor_max:
                    if self.regularization_factor == 0.0:
                        self.regularization_factor = opts.regularization_factor_min
                    else:
                        self.regularization_factor *= opts.regularization_factor_increase
                    self.H_k += self.regularization_factor*cs.DM.eye(self.nx)
                print('factor', self.regularization_factor)
            else:
                self.H_k = iterate.hessian_lagrangian_k

    def prepare_subproblem_bounds_constraints(self,
                                              iterate: Iterate,
                                              problem: NLPProblem):
        """
        Prepare the bounds for the constraints in the LP.
        I.e., min_{x}   g_k.T x
              s.t.  lba <= A_k*x <= uba,
                    lbx <= x <= ubx.
        Linearizing the constraints gives a constant part that needs to be
        added to the bounds.
        """
        self.lba_k = cs.vertcat(problem.lbg - iterate.g_k)
        self.uba_k = cs.vertcat(problem.ubg - iterate.g_k)

    def prepare_subproblem_bounds_variables(self,
                                            tr: TrustRegion,
                                            iterate: Iterate,
                                            problem: NLPProblem):
        """
        Prepare the bounds for the variables in the LP.
        I.e., min_{x}   g_k.T x
              s.t.  lba <= A_k*x <= uba,
                    lbx <= x <= ubx.
        The constant part of the linearization needs to be added in the bounds 
        and the trust-region needs to be taken into account as well.
        """
        self.lbd_k = cs.fmax(-tr.tr_radius_k*tr.tr_scale_mat_inv_k @
                             cs.DM.ones(self.nx, 1), problem.lbx - iterate.x_k)
        self.ubd_k = cs.fmin(tr.tr_radius_k*tr.tr_scale_mat_inv_k @
                             cs.DM.ones(self.nx, 1), problem.ubx - iterate.x_k)

    