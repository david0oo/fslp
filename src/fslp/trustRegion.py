"""
This file is used for the trust-region update in the solver
"""
from __future__ import annotations # <-still need this.
from typing import TYPE_CHECKING
import casadi as cs
if TYPE_CHECKING:
    from fslp.options import Options
    from fslp.direction import Direction
    from fslp.iterate import Iterate
    from fslp.nlp_problem import NLPProblem
    from fslp.logger import Logger


class TrustRegion:

    def __init__(self, options: Options):
        
        self.use_sqp = options.use_sqp
        self.f_corrected_iterate = None
        self.tr_ratio = 0.0
        self.model_of_objective = 0.0

        self.tr_radius_k = options.tr_radius0
        self.tr_scale_mat_k = options.tr_scale_mat0
        self.tr_scale_mat_inv_k = options.tr_scale_mat_inv0
        self.tr_reduction_alpha = 0.5 # Remove hard-coded stuff .....
        self.step_accepted = False

    # def eval_m_k(self, p:cs.DM):
    #     """
    #     In case of SQP:
    #     Evaluates the quadratic model of the objective function, i.e.,
    #     m_k(x_k; p) = grad_f_k.T @ p + p.T @ H_k @ p
    #     H_k denotes the Hessian Approximation

    #     In case of SLP:
    #     Evaluates the linear model of the objective function, i.e.,
    #     m_k(x_k; p) = grad_f_k.T @ p. This model is used in the termination
    #     criterion.

    #     Args:
    #         p (Casadi DM vector): the search direction where the linear model
    #                               should be evaluated

    #     Returns:
    #         double: the value of the linear model at the given direction p.
    #     """
    #     if self.use_sqp:
    #         return self.val_grad_f_k.T @ p + 0.5 * p.T @ self.H_k @ p
    #     else:
    #         return self.val_grad_f_k.T @ p
        
    def evaluate_predicted_reduction(self, model_of_objective):
        return -model_of_objective
    
    def evaluate_actual_reduction(self, f_old: float, f_correction: float):
        return f_old - f_correction

    def eval_trust_region_ratio(self, iterate: Iterate, direction: Direction, problem: NLPProblem, log: Logger):
        """
        We evaluate the trust region ratio here.

        rho = (f(x_k) - f(x_k + p_k_correction)) / (-m_k(p_k))

        x_k is the current iterate
        p_k is the solution of the original QP
        p_k_correction comes from the feasibility iterations
        """
        f_correction = problem.eval_f(iterate.x_inner_iterates, iterate.p, log)
        log.list_mks.append(float(cs.fabs(direction.m_k)))
        if (iterate.f_k - f_correction) <= 0:
            print('ared is negative')
        self.tr_ratio = (iterate.f_k - f_correction)/(-direction.m_k)

    def tr_reduction(self, direction: Direction, opts: Options):
        """
        Reduce the trust-region radius
        """
        self.tr_radius_k = self.tr_reduction_alpha * cs.norm_inf(self.tr_scale_mat_k @ direction.d_k)

    def tr_update(self, direction: Direction, opts: Options):
        """
        We use the Trust-Region Update of the paper Wright, Tenny 
        'A feasible Trust-Region SQP Algorithm' to update the trust-region
        radius.
        """
        # Adjust the trust-region radius
        if self.tr_ratio < opts.tr_eta1:
            self.tr_radius_k = opts.tr_alpha1 * \
                cs.norm_inf(self.tr_scale_mat_k @ direction.d_k)
        elif self.tr_ratio > opts.tr_eta2 and\
                cs.fabs(cs.norm_inf(
                self.tr_scale_mat_k @ direction.d_k) - self.tr_radius_k) < opts.tr_tol:
            self.tr_radius_k = cs.fmin(
                opts.tr_alpha2*self.tr_radius_k, opts.tr_radius_max)
        # else: keep the trust-region radius

    def step_acceptable(self, opts: Options):
        """
        This function performs the step acceptance procedure.

        Returns:
            bool: variable that says if step was accepted or not.
        """
        # Update the decision variables and multipliers if step accepted
        if self.tr_ratio > opts.tr_acceptance:
            # self.x_k = self.x_k_correction
            # self.lam_x_k = self.lam_p_x_k
            # self.lam_g_k = self.lam_p_g_k
            return True # step acceptable
        else:
            return False # step not acceptable
