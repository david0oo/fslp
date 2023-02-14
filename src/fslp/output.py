"""
This file handles the output of the Feasible Sequential Linear/Quadratic
Programming solver
"""
import casadi as cs
import numpy as np

class Output:

    def __init__(self) -> None:
        pass

    def print_iteration_header(self):
        """ 
        Prints the iteration header.
        """
        print(("{iter:>6} | {m_k:^10} | {grad_lag:^10} | {feas:^10} | "
                   "{compl:^10} | {f_x:^10} | {lam_g:^13} | {lam_x:^13} | "
                   "{feas_it:^9} | {tr_rad:^13} | {tr_ratio:^10}").format(
                        iter='iter', m_k='m_k', grad_lag='grad_lag',
                        compl='compl', f_x='f(x)', lam_g='||lamg||_inf',
                        lam_x='||lamx||_inf', feas='infeas',
                        feas_it='feas iter', tr_rad='tr_rad',
                        tr_ratio='tr_ratio'))

    def print_output(self, i:int):
        """
        This function prints the iteration output to the console.

        Args:
            i (integer): the iteration index of the solve operator.
        """
        if i % 10:
            self.print_iteration_header()

        print(("{iter:>6} | {m_k:^10.4e} | {grad_lag:^10.4e} | {feas:^10.4e} | "
               "{compl:^10.4e} | {f_x:^10.4f} | {lam_g:^13.4e} | "
               "{lam_x:^13.4e} | {feas_it:^9} | {tr_rad:^13.5e} | "
               "{tr_ratio:^10.8f}").format(
                     iter=i, m_k=np.array(self.m_k).squeeze(),
                     grad_lag=np.array(cs.norm_inf(
                         self.__eval_grad_lag(
                             self.x_k, self.lam_g_k, self.lam_x_k))).squeeze(),
                     compl=np.array(
                         self.__complementarity_condition()).squeeze(),
                     f_x=np.array(self.val_f_k).squeeze(),
                     lam_g=np.array(cs.norm_inf(self.lam_g_k)).squeeze(),
                     lam_x=np.array(cs.norm_inf(self.lam_x_k)).squeeze(),
                     feas=self.feasibility_measure(self.x_k, self.val_g_k),
                     feas_it=self.feas_iter,
                     tr_rad=np.array(self.tr_radii[-1]).squeeze(),
                     tr_ratio=np.array(self.rho_k).squeeze()))
