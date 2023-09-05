""" 
This class determines what subproblem is solved. Either QP or LP
"""
# Import standard libraries
from __future__ import annotations # <-still need this.
from typing import TYPE_CHECKING
import casadi as cs
import numpy as np
# Import self-written libraries
if TYPE_CHECKING:
    from .options import Options
    from .nlp_problem import NLPProblem
    from .iterate import Iterate


class Subproblem:

    def __init__(self,
                 problem: NLPProblem,
                 parameters: Options,
                 iterate: Iterate):
        self.use_sqp = parameters.use_sqp

        # Set up subproblem solver
        # self.subproblem_sol_opts["dump_in"] = True
        # self.subproblem_sol_opts["dump_out"] = True
        # self.subproblem_sol_opts["dump"] = True
        A_placeholder = problem.jacobian_g_function(iterate.x_k, iterate.p)
        if self.use_sqp:
            B_placeholder = problem.hessian_lagrangian_function(iterate.x_k,
                                                                iterate.p,
                                                                1.0,
                                                                iterate.lam_g_k)

            qp_struct = {   'h': B_placeholder.sparsity(),
                            'a': A_placeholder.sparsity()}
            
            # self.subproblem_sol_opts["print_out"] = True
            # qp_struct["h"].to_file("h.mtx")
            # qp_struct["a"].to_file("a.mtx")
            # print(self.subproblem_sol_opts)
            self.subproblem_solver = cs.conic(  "qpsolver",
                                                parameters.subproblem_solver_name,
                                                qp_struct,
                                                parameters.subproblem_solver_opts)
        else:
            lp_struct = {'a': A_placeholder.sparsity()}
            
            self.subproblem_solver = cs.conic(  "lpsolver",
                                                parameters.subproblem_solver_name,
                                                lp_struct,
                                                parameters.subproblem_solver_opts)

    # def __create_subproblem_solver(self):
    #     """
    #     This function creates an LP-solver object with the casadi conic 
    #     operation.
    #     """
    #     # self.subproblem_sol_opts["dump_in"] = True
    #     # self.subproblem_sol_opts["dump_out"] = True
    #     # self.subproblem_sol_opts["dump"] = True
    #     if self.use_sqp:
    #         B_placeholder = self.hess_lag_fun(self.x0, self.lam_g0, self.lam_x0)

    #         qp_struct = {   'h': B_placeholder.sparsity(),
    #                         'a': self.A_k.sparsity()}
            
    #         # self.subproblem_sol_opts["print_out"] = True

    #         # qp_struct["h"].to_file("h.mtx")
    #         # qp_struct["a"].to_file("a.mtx")
    #         # print(self.subproblem_sol_opts)

    #         self.subproblem_solver = cs.conic(  "qpsol",
    #                                             self.subproblem_sol,
    #                                             qp_struct,
    #                                             self.subproblem_sol_opts)
    #     else:
    #         lp_struct = {'a': self.A_k.sparsity()}
            
    #         self.subproblem_solver = cs.conic(  "qpsol",
    #                                             self.subproblem_sol,
    #                                             lp_struct,
    #                                             self.subproblem_sol_opts)

    # def solve_lp(self,
    #              g:cs.DM=None,
    #              a:cs.DM=None,
    #              lba:cs.DM=None,
    #              uba:cs.DM=None,
    #              lbx:cs.DM=None,
    #              ubx:cs.DM=None,
    #              x0:cs.DM=None,
    #              lam_x0:cs.DM=None,
    #              lam_a0:cs.DM=None):
    #     """
    #     This function solves the lp subproblem. Additionally some processing
    #     of the result is done and the statistics are saved. The input signature
    #     is the same as for a casadi lp solver.

    #     Args:
    #         g (Casadi DM vector, optional): Objective vector. Defaults to None.
    #         a (Casadi DM array, optional): Constraint matrix. Defaults to None.
    #         lba (Casadi DM vector, optional): Lower bounds on constraint 
    #                                           matrix. Defaults to None.
    #         uba (Casadi DM vector, optional): Upper bounds on constraint
    #                                           matrix. Defaults to None.
    #         lbx (Casadi DM vector, optional): Lower bounds on states. 
    #                                           Defaults to None.
    #         ubx (Casadi DM vector, optional): Upper bounds on states.
    #                                           Defaults to None.

    #     Returns:
    #         (bool, Casadi DM vector, Casadi DM vector): First indicates of LP
    #         was solved succesfully, second entry is the new search direction, 
    #         the third entry are the lagrange multipliers for the new
    #         search direction
    #     """
    #     res = self.subproblem_solver(   g=g,
    #                                     a=a,
    #                                     lba=lba,
    #                                     uba=uba,
    #                                     lbx=lbx,
    #                                     ubx=ubx,
    #                                     x0=x0,
    #                                     lam_x0=lam_x0,
    #                                     lam_a0=lam_a0)

    #     # Keep track that bounds of QP are guaranteed. If not because of a
    #     # tolerance, make them exact.
    #     p_tmp = res['x']
    #     # Get indeces where variables are violated
    #     # lower_p = list(np.nonzero(np.array(p_tmp < lbx).squeeze())[0])
    #     # upper_p = list(np.nonzero(np.array(p_tmp > ubx).squeeze())[0])

    #     # # Resolve the 'violation' in the search direction
    #     # if bool(lower_p):
    #     #     p_tmp[lower_p] = lbx[lower_p]
    #     # if bool(upper_p):
    #     #     p_tmp[upper_p] = ubx[upper_p]

    #     # # Process the new search directions and multipliers w/o slacks
    #     # p = p_tmp
    #     lam_p_g = res['lam_a']
    #     lam_p_x = res['lam_x']

    #     solve_success = self.subproblem_solver.stats()['success']

    #     return (solve_success, p, lam_p_g, lam_p_x)

    # def solve_qp(self,
    #              h:cs.DM=None,
    #              g:cs.DM=None,
    #              a:cs.DM=None,
    #              lba:cs.DM=None,
    #              uba:cs.DM=None,
    #              lbx:cs.DM=None,
    #              ubx:cs.DM=None,
    #              x0:cs.DM=None,
    #              lam_x0:cs.DM=None,
    #              lam_a0:cs.DM=None):
    #     """
    #     This function solves the qp subproblem. Additionally some processing of
    #     the result is done and the statistics are saved. The input signature is
    #     the same as for a casadi qp solver.
    #     Input:
    #     h       Matrix in QP objective
    #     g       Vector in QP objective
    #     a       Matrix for QP constraints
    #     lba     lower bounds of constraints
    #     uba     upper bounds of constraints
    #     lbx     lower bounds of variables
    #     ubx     upper bounds of variables

    #     Return:
    #     solve_success   Bool, indicating if qp was succesfully solved
    #     p_scale         Casadi DM vector, the new search direction
    #     lam_p_scale     Casadi DM vector, the lagrange multipliers for the new
    #                     search direction
    #     """
    #     res = self.subproblem_solver(   h=h,
    #                                     g=g,
    #                                     a=a,
    #                                     lba=lba,
    #                                     uba=uba,
    #                                     lbx=lbx,
    #                                     ubx=ubx,
    #                                     x0=x0,
    #                                     lam_x0=lam_x0,
    #                                     lam_a0=lam_a0)
        
    #     # Save some statistics

    #     # Keep track that bounds of QP are guaranteed. If not because of a 
    #     # tolerance, make them exact.

    #     p = res['x']
    #     # Get indeces where variables are violated
    #     # lower_p = list(np.nonzero(np.array(p_tmp < lbx).squeeze())[0])
    #     # upper_p = list(np.nonzero(np.array(p_tmp > ubx).squeeze())[0])

    #     # # Resolve the 'violation' in the search direction
    #     # if bool(lower_p):
    #     #     p_tmp[lower_p] = lbx[lower_p]
    #     # if bool(upper_p):
    #     #     p_tmp[upper_p] = ubx[upper_p]

    #     # # Process the new search directions and multipliers w/o slacks
    #     # p = p_tmp
    #     lam_p_g = res['lam_a']
    #     lam_p_x = res['lam_x']
        
    #     solve_success = self.subproblem_solver.stats()['success']

    #     return (solve_success, p, lam_p_g, lam_p_x)

    # def solve_subproblem(   self,
    #                         g:cs.DM=None,
    #                         lba:cs.DM=None,
    #                         uba:cs.DM=None,
    #                         lbx:cs.DM=None,
    #                         ubx:cs.DM=None,
    #                         x0:cs.DM=None,
    #                         lam_x0:cs.DM=None,
    #                         lam_a0:cs.DM=None):
    def solve_subproblem(   self, problem_dict: dict):
        """
        This function solves the qp subproblem. Additionally some processing of
        the result is done and the statistics are saved. The input signature is
        the same as for a casadi qp solver.
        Input:
        g       Vector in QP objective
        lba     lower bounds of constraints
        uba     upper bounds of constraints
        lbx     lower bounds of variables
        ubx     upper bounds of variables

        Return:
        solve_success   Bool, indicating if subproblem was succesfully solved
        p_scale         Casadi DM vector, the new search direction
        lam_p_scale     Casadi DM vector, the lagrange multipliers for the new
                        search direction
        """
        print("Subproblem is:")
        print(problem_dict)

        res = self.subproblem_solver(**problem_dict)

        p = res['x']
        # Get indeces where variables are violated
        # lower_p = list(np.nonzero(np.array(p_tmp < lbx).squeeze())[0])
        # upper_p = list(np.nonzero(np.array(p_tmp > ubx).squeeze())[0])

        # # Resolve the 'violation' in the search direction
        # if bool(lower_p):
        #     p_tmp[lower_p] = lbx[lower_p]
        # if bool(upper_p):
        #     p_tmp[upper_p] = ubx[upper_p]

        # # Process the new search directions and multipliers w/o slacks
        # p = p_tmp
        lam_p_g = res['lam_a']
        lam_p_x = res['lam_x']
        
        solve_success = self.subproblem_solver.stats()['success']

        return (solve_success, p, lam_p_g, lam_p_x)

        # if self.use_sqp:
        #     return self.solve_qp(   h=self.H_k,
        #                             g=g,
        #                             a=self.A_k,
        #                             lba=lba,
        #                             uba=uba,
        #                             lbx=lbx,
        #                             ubx=ubx,
        #                             x0=x0,
        #                             lam_x0=lam_x0,
        #                             lam_a0=lam_a0)
        # else:
        #     return self.solve_lp(   g=g,
        #                             a=self.A_k,
        #                             lba=lba,
        #                             uba=uba,
        #                             lbx=lbx,
        #                             ubx=ubx,
        #                             x0=x0,
        #                             lam_x0=lam_x0,
        #                             lam_a0=lam_a0)