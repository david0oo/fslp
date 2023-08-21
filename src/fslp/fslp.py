"""
We provide a prototypical implementation of the FSLP method.
"""
import casadi as cs
import numpy as np
from timeit import default_timer as timer

from .nlp_problem import NLPProblem
from .options import Options
from .logger import Logger
from .output import Output
from .direction import Direction
from .iterate import Iterate
from .trustRegion import TrustRegion
from .subproblem import Subproblem

cs.DM.set_precision(16)


class FSLP_Method:

    def __init__(self, nlp_dict: dict, initialization_dict: dict, opts: dict):
        """
        The constructor. More variables are set and initialized in specific
        functions below.
        """
        self.nlp_problem = NLPProblem(nlp_dict, initialization_dict, opts)
        self.options = Options(opts)
        self.logger = Logger()
        self.output = Output()
        self.inner_direction = Direction()
        self.outer_direction = Direction()
        self.iterate = Iterate()
        self.trust_region = TrustRegion()
        self.subproblem = Subproblem()

        self.subproblem_solver = None
        self.solver_type = 'SLP'
        self.regularization_factor_min = 1e-4

    def solve(self, init_dict: dict):
        """
        Solve NLP with feasible SQP method.

        Args:
            problem_dict (dict): dictionary of the problem specification.
            init_dict (dict): dictionary of initialization specifications.
            opts (dict, optional): dictionary of options. Defaults to {}.
            callback (function, optional): A callback function.
                                           Defaults to None.

        Raises:
            ValueError: init_dict is empty
            ValueError: problem_dict is empty

        Returns:
            Casadi DM vector, double: the optimal argument of solution and the
                                      optimal function value of solution.
        """
        # scalene_profiler.start()

        self.lasttime = timer()

        self.__initialize_parameters(problem_dict, init_dict, opts)
        self.__initialize_functions(opts)
        self.__initialize_stats()
        
        self.iterate.__initialize(init_dict, )
        # self.x_k = self.x0
        # self.lam_g_k = self.lam_g0
        # self.lam_x_k = self.lam_x0
        # self.__eval_grad_jac()

        if self.iterate.feasibility_measure > self.options.feas_tol:
            raise ValueError('Initial guess needs to be feasible!!')
        # if self.feasibility_measure(self.x_k, self.val_g_k) > self.feas_tol:
        #     raise ValueError('Initial guess needs to be feasible!!')

        self.slacks_zero = False
        self.slacks_zero_iter = 0
        self.slacks_zero_n_eval_g = 0

        self.step_accepted = False

        self.tr_scale_mat_inv_k = self.tr_scale_mat_inv0
        self.tr_scale_mat_k = self.tr_scale_mat0
        self.tr_rad_k = self.tr_rad0
        self.rho_k = 0

        self.success = False

        self.lam_tmp_g = self.lam_g0
        self.lam_tmp_x = self.lam_x0


        self.inner_iter_count = 0
        self.inner_iter_limit_count = 0

        self.__prepare_subproblem_matrices()
        self.__prepare_subproblem_bounds_variables()
        self.__prepare_subproblem_bounds_constraints()

        self.__create_subproblem_solver()

        self.feas_iter = -1
        self.m_k = -1
        self.p_k = cs.DM.zeros(self.nx)

        self.tr_radii = [self.tr_rad_k]
        self.inner_iters = []
        self.inner_feas = []
        self.inner_as_exac = []
        self.inner_kappas = []
        self.inner_steps = []
        self.asymptotic_exactness = []
        self.list_iter = [self.x_k]
        self.list_feas = [self.feasibility_measure(self.x_k, self.val_g_k)]
        self.list_times = [0.0]
        self.list_mks = []
        self.list_grad_lag = []

        self.n_iter = 0
        self.__check_slacks_zero()
        self.accepted_counter = 0

        # self.x_k.to_file('x0.mtx')
        # self.val_g_k.to_file('g0.mtx')
        # self.val_grad_f_k.to_file('gf0.mtx')

        for i in range(self.max_iter):
            # Do the FSLP outer iterations here
            # print iteration here
            if self.verbose:
                self.print_output(i)

            self.n_iter = i + 1
            self.lam_tmp_g = self.lam_g_k
            self.lam_tmp_x = self.lam_x_k

            self.tr_radii.append(self.tr_rad_k)

            # self.g_k.to_file('gf.mtx')
            # self.lb_var_k.to_file('lb_var.mtx')
            # self.ub_var_k.to_file('ub_var.mtx')
            # self.lba_k.to_file('lba.mtx')
            # self.uba_k.to_file('uba.mtx')
            # if self.use_sqp:
            #     self.H_k.to_file('Bk.mtx')
            # self.A_k.to_file('Jk.mtx')
            
            (solve_success,
            self.p_k,
            self.lam_p_g_k,
            self.lam_p_x_k) = self.solve_subproblem(    g=self.g_k,
                                                        lba=self.lba_k,
                                                        uba=self.uba_k,
                                                        lbx=self.lb_var_k,
                                                        ubx=self.ub_var_k,
                                                        x0=self.p_k,
                                                        lam_x0=self.lam_x_k,
                                                        lam_a0=self.lam_g_k)

            
            if not solve_success:
                if self.use_sqp:
                    # TODO: correct this
                    print('Something went wrong in QP: ', self.subproblem_solver.stats()[
                        'return_status'])
                else:
                    print('Something went wrong in LP: ', self.subproblem_solver.stats()[
                        'return_status'])
                break

            # self.p_k = self.__set_optimal_slack_step(self.x_k, self.p_k)
            # self.p_k.to_file('dx.mtx')
            self.m_k = self.eval_m_k(self.p_k)

            if cs.fabs(self.m_k) < self.optim_tol:
                if self.verbose:
                    print('Optimal Point Found? Linear model is {m_k:^10.4e}.'.format(m_k=np.array(self.m_k).squeeze()))
                self.success = True
                self.stats['success'] = True
                self.list_mks.append(float(cs.fabs(self.m_k)))
                break

            self.step_inf_norm = cs.norm_inf(self.tr_scale_mat_k @ self.p_k)

            self.prev_step_inf_norm = self.step_inf_norm
            (self.x_k_correction,
                self.lam_p_g_k,
                self.lam_p_x_k,
                self.feas_iter) = self.feasibility_iterations(self.p_k)

            if not self.kappa_acceptance:
                step_accepted = False
                if self.verbose:
                    print('Rejected inner iterates or asymptotic exactness')
                self.tr_rad_k = 0.5*cs.norm_inf(self.tr_scale_mat_k @ self.p_k)
            else:
                self.eval_trust_region_ratio()
                self.tr_update()
                step_accepted = self.step_update()

            if step_accepted:
                self.list_grad_lag.append(
                    np.array(cs.norm_inf(
                        self.grad_lag_fun(
                            self.x_k, self.lam_g_k, self.lam_x_k))).squeeze())
                self.list_iter.append(self.x_k)
                if self.verbose:
                    print('ACCEPTED')
                self.__eval_grad_jac(step_accepted)
                self.__prepare_subproblem_matrices()
                self.__prepare_subproblem_bounds_constraints()
                self.accepted_counter += 1
                self.__check_slacks_zero()
                self.list_feas.append(self.feasibility_measure(self.x_k,
                                                               self.val_g_k))
                self.list_times.append(timer() - self.lasttime)
            else:
                if self.verbose:
                    print('REJECTED')

            self.__prepare_subproblem_bounds_variables()

        self.stats['t_wall'] = timer() - self.lasttime
        if hasattr(self, 'slacks_zero_t_wall'):
            self.stats['t_wall_zero_slacks'] = self.slacks_zero_t_wall
        else:
            self.stats['t_wall_zero_slacks'] = self.stats['t_wall']
        self.stats['iter_count'] = self.n_iter
        self.stats['accepted_iter'] = self.accepted_counter
        self.stats['iter_slacks_zero'] = self.slacks_zero_iter
        self.stats['n_eval_g_slacks_zero'] = self.slacks_zero_n_eval_g
        return self.x_k, self.val_f_k



    def feasibility_iterations(self, p:cs.DM):
        """
        The feasibility iterations are performed here.

        Args:
            p (Casadi DM vector): the search direction given from outer
                                  iterations

        Returns:
            Casadi DM vector, Casadi DM vector, Casadi DM vector, integer:
            a tuple that returns the new temporary state x_tmp, for that temp
            state the multipliers for constraints and states, and the number of
            inner iterations to achieve feasibility.
        """
        p_tmp = p
        lam_p_g_tmp = self.lam_p_g_k
        lam_p_x_tmp = self.lam_p_x_k

        if self.use_anderson:
            self.anderson_acc_init_memory(p_tmp, self.x_k)

        self.x_tmp = self.x_k + p_tmp
        self.g_tmp = self.__eval_g(self.x_tmp)

        self.kappa_acceptance = False

        inner_iter = 0
        asymptotic_exactness = []
        as_exac = cs.norm_2(
                self.p_k - self.x_tmp + self.x_k) / cs.norm_2(self.p_k)
        self.prev_infeas = self.feasibility_measure(self.x_tmp, self.g_tmp)
        self.curr_infeas = self.feasibility_measure(self.x_tmp, self.g_tmp)
        feasibilities = [self.prev_infeas]
        step_norms = []
        kappas = []

        watchdog_prev_inf_norm = self.prev_step_inf_norm
        accumulated_as_ex = 0

        for j in range(self.max_inner_iter):

            # if cs.norm_inf(self.tr_scale_mat_k @ p_tmp) < self.feas_tol:
            if self.curr_infeas < self.feas_tol:
                inner_iter = j
                if as_exac < 0.5:               
                    self.kappa_acceptance = True
                    break
                else:
                    self.kappa_acceptance = False
                    break
            elif j > 0 and (self.curr_infeas > 1.0 or as_exac > 1.0):
                self.kappa_acceptance = False
                break

            inner_iter = j+1
            self.lam_tmp_g = self.lam_p_g_k
            self.lam_tmp_x = self.lam_p_x_k

            if self.gradient_correction:
                grad_L_tmp = self.__eval_grad_lag(self.x_tmp, lam_p_g_tmp, lam_p_x_tmp)
                print('Gradient of Lagrangian: ', cs.norm_inf(grad_L_tmp))
                # Do the gradient correction, could also be + instead of -??
                grad_f_correction = grad_L_tmp - self.A_k.T @ lam_p_g_tmp - lam_p_x_tmp
            else:
                # Do just Zero-Order Iterations
                if self.use_sqp:
                    grad_f_correction = self.val_grad_f_k + self.H_k @ (self.x_tmp - self.x_k)
                else:
                    grad_f_correction = self.val_grad_f_k
            
            lb_var_correction = cs.fmax(-self.tr_rad_k*self.tr_scale_mat_inv_k @
                          cs.DM.ones(self.nx, 1) - self.x_tmp+self.x_k,
                          self.lbx - self.x_tmp)
            ub_var_correction = cs.fmin(self.tr_rad_k*self.tr_scale_mat_inv_k @
                          cs.DM.ones(self.nx, 1) - self.x_tmp+self.x_k,
                          self.ubx - self.x_tmp)

            lba_correction = self.lbg - self.g_tmp
            uba_correction = self.ubg - self.g_tmp

            # grad_f_correction.to_file('gf_feas.mtx')
            # lba_correction.to_file('lba_correction.mtx')
            # uba_correction.to_file('uba_correction.mtx')
            # lb_var_correction.to_file('lb_var_correction.mtx')
            # ub_var_correction.to_file('ub_var_correction.mtx')
            
            (_,
            p_tmp,
            lam_p_g_tmp,
            lam_p_x_tmp) = self.solve_subproblem(   g=grad_f_correction,
                                                    lba=lba_correction,
                                                    uba=uba_correction,
                                                    lbx=lb_var_correction,
                                                    ubx=ub_var_correction,
                                                    x0=p_tmp,
                                                    lam_x0=lam_p_x_tmp,
                                                    lam_a0=lam_p_g_tmp)

            # p_tmp.to_file('dx_feas.mtx')

            self.step_inf_norm = cs.norm_inf(self.tr_scale_mat_k @ p_tmp)


            if self.use_anderson:
                self.x_tmp = self.anderson_acc_step_update(p_tmp, self.x_tmp, j+1)
            else:
                self.x_tmp = self.x_tmp + p_tmp
            self.g_tmp = self.__eval_g(self.x_tmp)  # x_tmp = x_{tmp-1} + p_tmp

            self.curr_infeas = self.feasibility_measure(self.x_tmp, self.g_tmp)
            self.prev_infeas = self.curr_infeas

            kappa = self.step_inf_norm/self.prev_step_inf_norm
            kappas.append(kappa)

            # self.p_k.to_file('p_k.mtx')
            # self.x_tmp.to_file('x_tmp.mtx')
            # self.x_k.to_file('x_k.mtx')

            as_exac = cs.norm_2(
                self.p_k - (self.x_tmp - self.x_k)) / cs.norm_2(self.p_k)
            if self.verbose:
                print("Kappa: ", kappa,
                      "Infeasibility", self.feasibility_measure(
                                self.x_tmp, self.g_tmp),
                      "Asymptotic Exactness: ", as_exac)

            # +1 excludes the first iteration from the kappa test
            accumulated_as_ex += as_exac
            if inner_iter % self.watchdog == 0:
                kappa_watch = self.step_inf_norm/watchdog_prev_inf_norm
                watchdog_prev_inf_norm = self.step_inf_norm
                if self.verbose:
                    print("kappa watchdog: ", kappa_watch)
                if self.curr_infeas < self.feas_tol and as_exac < 0.5:
                    self.kappa_acceptance = True
                    break
                if kappa_watch > self.contraction_acceptance or\
                        accumulated_as_ex/self.watchdog > 0.5:
                    self.kappa_acceptance = False
                    break
                accumulated_as_ex = 0

            feasibilities.append(
                self.feasibility_measure(self.x_tmp, self.g_tmp))
            asymptotic_exactness.append(as_exac)
            step_norms.append(cs.norm_inf(p_tmp))

            self.prev_step_inf_norm = self.step_inf_norm
            self.lam_tmp_g = self.lam_p_g_k
            self.lam_tmp_x = self.lam_p_x_k

        self.stats['inner_iter'] += inner_iter
        self.inner_iters.append(inner_iter)
        self.inner_steps.append(step_norms)
        self.inner_feas.append(feasibilities)
        self.inner_as_exac.append(asymptotic_exactness)
        self.inner_kappas.append(kappas)
        self.asymptotic_exactness.append(as_exac)

        return self.x_tmp, lam_p_g_tmp, lam_p_x_tmp, inner_iter
