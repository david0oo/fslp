"""
This file is used for the trust-region update in the solver
"""
import casadi as cs

class TrustRegion:

    def __init__(self) -> None:
        pass

    def eval_trust_region_ratio(self):
        """
        We evaluate the trust region ratio here.

        rho = (f(x_k) - f(x_k + p_k_correction)) / (-m_k(p_k))

        x_k is the current iterate
        p_k is the solution of the original QP
        p_k_correction comes from the feasibility iterations
        """
        f_correction = self.f_fun(self.x_k_correction)
        self.list_mks.append(float(cs.fabs(self.m_k)))
        if (self.val_f_k - f_correction) <= 0:
            if self.verbose:
                print('ared is negative')
        self.rho_k = (self.val_f_k - f_correction)/(-self.m_k)

    def tr_update(self):
        """
        We use the Trust-Region Update of the paper Wright, Tenny 
        'A feasible Trust-Region SQP Algorithm' to update the trust-region
        radius.
        """
        # Adjust the trust-region radius
        if self.rho_k < self.tr_eta1:
            self.tr_rad_k = self.tr_alpha1 * \
                cs.norm_inf(self.tr_scale_mat_k @ self.p_k)
        elif self.rho_k > self.tr_eta2 and\
                cs.fabs(cs.norm_inf(
                self.tr_scale_mat_k @ self.p_k) - self.tr_rad_k) < self.tr_tol:
            self.tr_rad_k = cs.fmin(
                self.tr_alpha2*self.tr_rad_k, self.tr_rad_max)
        # else: keep the trust-region radius

    def step_update(self):
        """
        This function performs the step acceptance procedure.

        Returns:
            bool: variable that says if step was accepted or not.
        """
        # Update the decision variables and multipliers if step accepted
        if self.rho_k > self.tr_acceptance:
            self.x_k = self.x_k_correction
            self.lam_x_k = self.lam_p_x_k
            self.lam_g_k = self.lam_p_g_k
            step_accepted = True
        else:
            step_accepted = False

        return step_accepted