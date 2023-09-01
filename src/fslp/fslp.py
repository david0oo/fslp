"""
We provide a prototypical implementation of the FSLP method.
"""
# Import standard libraries
import casadi as cs
import numpy as np
from timeit import default_timer as timer
# Import self-written libraries
from fslp.nlp_problem import NLPProblem
from fslp.options import Options
from fslp.logger import Logger
from fslp.output import Output
from fslp.iterate import Iterate
from fslp.direction import Direction
from fslp.trustRegion import TrustRegion
from fslp.subproblem import Subproblem

cs.DM.set_precision(16)


class FSLP:

    def __init__(self, nlp_dict: dict, opts: dict):
        """
        The constructor. More variables are set and initialized in specific
        functions below.
        """
        # First setup the options properly
        self.options = Options(nlp_dict, opts)
        # Setup NLP
        self.nlp_problem = NLPProblem(nlp_dict, self.options)
        self.log = Logger()
        self.output = Output()

        # use options to initialize everything properly

        # self.inner_direction = Direction(self.nlp_problem, self.options)
        self.direction = Direction(self.nlp_problem, self.options)

        self.iterate = Iterate(self.nlp_problem)

        self.trust_region = TrustRegion(self.options)
        self.subproblem = Subproblem(self.nlp_problem, self.options, self.iterate)

        # self.subproblem_solver = None
        # self.solver_type = 'SLP'

    def __call__(self, init_dict: dict):
        """
        Solve NLP with feasible SLP/SQP method.

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

        self.output.print_header()

        # self.__initialize_parameters(problem_dict, init_dict, opts)
        # self.__initialize_functions(opts)
        self.nlp_problem.initialize(init_dict)

        # self.__initialize_stats()
        self.log.reset()
        
        # self.__eval_grad_jac()
        self.iterate.initialize(init_dict, self.nlp_problem, self.options, self.log)

        if self.iterate.infeasibility > self.options.feasibility_tol:
            raise ValueError('Initial guess needs to be feasible!!')
        # if self.feasibility_measure(self.x_k, self.val_g_k) > self.feas_tol:
        #     raise ValueError('Initial guess needs to be feasible!!')

        self.slacks_zero = False
        self.slacks_zero_iter = 0
        self.slacks_zero_n_eval_g = 0

        self.trust_region.step_accepted = False

        # self.tr_scale_mat_inv_k = self.tr_scale_mat_inv0
        # self.tr_scale_mat_k = self.tr_scale_mat0
        # self.tr_rad_k = self.tr_rad0
        # self.rho_k = 0

        # self.success = False

        self.lam_tmp_g = self.iterate.lam_g_k
        self.lam_tmp_x = self.iterate.lam_x_k

        # self.inner_iter_count = 0
        # self.inner_iter_limit_count = 0

        self.direction.prepare_subproblem_matrices(self.iterate, self.options)
        self.direction.prepare_subproblem_bounds_variables(self.trust_region, self.iterate, self.nlp_problem)
        self.direction.prepare_subproblem_bounds_constraints(self.iterate, self.nlp_problem)

        # self.__create_subproblem_solver()

        self.log.feasibility_iteration_counter = -1
        self.direction.m_k = -1
        self.direction.d_k = cs.DM.zeros(self.nlp_problem.number_variables)

        # self.tr_radii = [self.tr_rad_k]
        # self.inner_iters = []
        # self.inner_feas = []
        # self.inner_as_exac = []
        # self.inner_kappas = []
        # self.inner_steps = []
        # self.asymptotic_exactness = []
        # self.list_iter = [self.x_k]
        # self.list_feas = [self.feasibility_measure(self.x_k, self.val_g_k)]
        # self.list_times = [0.0]
        # self.list_mks = []
        # self.list_grad_lag = []

        self.log.iteration_counter = 0
        self.log.tr_radii.append(self.trust_region.tr_radius_k)
        self.log.inner_iters.append(0)
        # self.__check_slacks_zero()
        self.log.accepted_iterations_counter = 0

        for i in range(self.options.max_iter):
            # Do the FSLP outer iterations here
            # print iteration here
            if self.options.verbose:
                self.output.print_output(i,
                                         self.iterate,
                                         self.direction,
                                         self.nlp_problem,
                                         self.trust_region,
                                         self.log)

            # self.n_iter = i + 1
            self.log.increment_iteration_counter()
            
            self.lam_tmp_g = self.iterate.lam_g_k
            self.lam_tmp_x = self.iterate.lam_x_k

            
            # Solve the LP / QP -----------------------------------------------
            solver_dict = {}
            solver_dict['g'] = self.direction.objective_vector_k
            solver_dict['a'] = self.direction.A_k
            solver_dict['lba'] = self.direction.lba_k
            solver_dict['uba'] = self.direction.uba_k
            solver_dict['lbx'] = self.direction.lbd_k
            solver_dict['ubx'] = self.direction.ubd_k
            solver_dict['x0'] = self.direction.d_k
            solver_dict['lam_x0'] = self.direction.lam_d_k
            solver_dict['lam_a0'] = self.direction.lam_a_k
            if self.options.use_sqp:
                solver_dict['h'] = self.direction.H_k

            (solve_success,
            self.direction.d_k,
            self.direction.lam_a_k,
            self.lam_d_k) = self.subproblem.solve_subproblem(solver_dict)
            # self.solve_subproblem(    g=self.g_k,
            #                                             lba=self.lba_k,
            #                                             uba=self.uba_k,
            #                                             lbx=self.lb_var_k,
            #                                             ubx=self.ub_var_k,
            #                                             x0=self.p_k,
            #                                             lam_x0=self.lam_x_k,
            #                                             lam_a0=self.lam_g_k)

            if not solve_success:
                error_string = 'Something went wrong in subproblem: ' + self.subproblem_solver.solve_message
                raise RuntimeError(error_string)

            self.direction.eval_m_k(self.iterate)

            if cs.fabs(self.direction.m_k) < self.options.optimality_tol:
                if self.options.verbose:
                    print('Optimal Point Found? Linear model is {m_k:^10.4e}.'.format(m_k=np.array(self.m_k).squeeze()))
                self.success = True
                self.log.solver_success = True
                self.log.list_mks.append(float(cs.fabs(self.direction.m_k)))
                break

            self.step_inf_norm = cs.norm_inf(self.trust_region.tr_scale_mat_k @ self.direction.d_k)

            # ----------------- feasibility iterations ------------------------

            self.prev_step_inf_norm = self.step_inf_norm
            (self.x_k_correction,
                self.lam_p_g_k,
                self.lam_p_x_k,
                self.feas_iter) = self.feasibility_iterations()

            # ------------- Trust-region iterations ---------------------------
            if not self.kappa_acceptance:
                step_accepted = False
                if self.options.verbose:
                    print('Rejected inner iterates or asymptotic exactness')
                self.tr_rad_k = 0.5*cs.norm_inf(self.trust_region.tr_scale_mat_k @ self.direction.d_k)
            else:
                self.trust_region.eval_trust_region_ratio(self.iterate, self.direction, self.nlp_problem, self.log)
                self.trust_region.tr_update(self.direction, self.options)
                step_accepted = self.trust_region.step_acceptable(self.options)

            self.iterate.step_update(step_accepted)

            # -------------- Step acceptance in trust-region iters ------------
            if step_accepted:
                self.log.list_grad_lag.append(
                    np.array(cs.norm_inf(
                        self.nlp_problem.gradient_lagrangian_function(
                            self.iterate.x_k, self.iterate.p, cs.DM([1]), self.iterate.lam_g_k, self.iterate.lam_x_k))).squeeze())
                self.log.list_iter.append(self.iterate.x_k)
                if self.options.verbose:
                    print('ACCEPTED')
                self.iterate.evaluate_quantities(self.nlp_problem, self.log, self.options)
                self.direction.prepare_subproblem_bounds_constraints(self.iterate, self.nlp_problem)
                self.direction.prepare_subproblem_matrices(self.iterate, self.options)
                
                # self.__eval_grad_jac(step_accepted)
                # self.__prepare_subproblem_matrices()
                # self.__prepare_subproblem_bounds_constraints()
                self.log.accepted_iterations_counter += 1
                # self.__check_slacks_zero()
                self.log.list_feas.append(self.iterate.infeasibility)
                self.log.list_times.append(timer() - self.lasttime)
            else:
                if self.options.verbose:
                    print('REJECTED')

            self.log.tr_radii.append(self.trust_region.tr_radius_k)
            self.direction.prepare_subproblem_bounds_variables(self.trust_region, self.iterate, self.nlp_problem)

        # Max_iter reached or optimal solution found
        self.stats['t_wall'] = timer() - self.lasttime
        if hasattr(self, 'slacks_zero_t_wall'):
            self.stats['t_wall_zero_slacks'] = self.slacks_zero_t_wall
        else:
            self.stats['t_wall_zero_slacks'] = self.stats['t_wall']
        self.stats['iter_count'] = self.n_iter
        self.stats['accepted_iter'] = self.accepted_counter
        self.stats['iter_slacks_zero'] = self.slacks_zero_iter
        self.stats['n_eval_g_slacks_zero'] = self.slacks_zero_n_eval_g

        return self.iterate.x_k, self.iterate.f_k, self.iterate.lam_g_k, self.iterate.lam_x_k

    def feasibility_iterations(self):
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
        self.direction.d_inner_iterates = self.direction.d_k
        self.direction.lam_a_inner_iterates = self.direction.lam_a_k
        self.direction.lam_d_inner_iterates = self.direction.lam_d_k

        if self.options.use_anderson:
            self.anderson_acceleration.init_memory(self.direction.d_inner_iterates, self.iterate.x_k)

        self.iterate.x_inner_iterates = self.iterate.x_k + self.direction.d_inner_iterates
        self.g_tmp = self.nlp_problem.eval_g(self.iterate.x_inner_iterates, self.iterate.p, self.log)

        self.kappa_acceptance = False

        inner_iter = 0
        asymptotic_exactness = []
        as_exac = cs.norm_2(
                self.direction.d_k - self.iterate.x_inner_iterates + self.iterate.x_k) / cs.norm_2(self.direction.d_k)
        self.prev_infeas = self.iterate.feasibility_measure(self.iterate.x_inner_iterates, self.g_tmp, self.nlp_problem)
        self.curr_infeas = self.iterate.feasibility_measure(self.iterate.x_inner_iterates, self.g_tmp, self.nlp_problem)
        feasibilities = [self.prev_infeas]
        step_norms = []
        kappas = []

        watchdog_prev_inf_norm = self.prev_step_inf_norm
        accumulated_as_ex = 0

        for j in range(self.options.max_inner_iter):

            # ----------- termination criterium -------------------------------
            # if cs.norm_inf(self.tr_scale_mat_k @ p_tmp) < self.feas_tol:
            if self.curr_infeas < self.options.feasibility_tol:
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


            # -------------- Prepare subproblem -------------------------------
            # if self.gradient_correction:
            #     grad_L_tmp = self.__eval_grad_lag(self.x_tmp, lam_p_g_tmp, lam_p_x_tmp)
            #     print('Gradient of Lagrangian: ', cs.norm_inf(grad_L_tmp))
            #     # Do the gradient correction, could also be + instead of -??
            #     grad_f_correction = grad_L_tmp - self.A_k.T @ lam_p_g_tmp - lam_p_x_tmp
            # else:
            # Do just Zero-Order Iterations
            if self.options.use_sqp:
                grad_f_correction = self.iterate.gradient_f_k + self.direction.H_k @ (self.iterate.x_inner_iterates - self.iterate.x_k)
            else:
                grad_f_correction = self.iterate.gradient_f_k
            
            lb_var_correction = cs.fmax(-self.trust_region.tr_radius_k*self.trust_region.tr_scale_mat_inv_k @
                          cs.DM.ones(self.nlp_problem.number_variables, 1) - self.iterate.x_inner_iterates+self.iterate.x_k,
                          self.nlp_problem.lbx - self.iterate.x_inner_iterates)
            ub_var_correction = cs.fmin(self.trust_region.tr_radius_k*self.trust_region.tr_scale_mat_inv_k @
                          cs.DM.ones(self.nlp_problem.number_variables, 1) - self.iterate.x_inner_iterates+self.iterate.x_k,
                          self.nlp_problem.ubx - self.iterate.x_inner_iterates)

            lba_correction = self.nlp_problem.lbg - self.g_tmp
            uba_correction = self.nlp_problem.ubg - self.g_tmp

            # ------------- Solve subproblem ---------------------------------
            solver_dict = {}
            solver_dict['g'] = grad_f_correction
            solver_dict['a'] = self.direction.A_k
            solver_dict['lba'] = lba_correction
            solver_dict['uba'] = uba_correction
            solver_dict['lbx'] = lb_var_correction
            solver_dict['ubx'] = ub_var_correction
            solver_dict['x0'] = self.direction.d_inner_iterates
            solver_dict['lam_x0'] = self.direction.lam_d_inner_iterates
            solver_dict['lam_a0'] = self.direction.lam_a_inner_iterates
            if self.options.use_sqp:
                solver_dict['h'] = self.direction.H_k
            
            (_,
            self.direction.d_inner_iterates,
            self.direction.lam_d_inner_iterates,
            self.direction.lam_a_inner_iterates) = self.subproblem.solve_subproblem(solver_dict)

            self.step_inf_norm = cs.norm_inf(self.trust_region.tr_scale_mat_k @ self.direction.d_inner_iterates)


            if self.options.use_anderson:
                self.iterate.x_inner_iterates = self.anderson_acceleration.step_update(self.direction.d_inner_iterates, self.iterate.x_inner_iterates, j+1)
            else:
                self.iterate.x_inner_iterates = self.iterate.x_inner_iterates + self.direction.d_inner_iterates
            self.g_tmp = self.nlp_problem.eval_g(self.iterate.x_inner_iterates,self.iterate.p, self.log)  # x_tmp = x_{tmp-1} + p_tmp


            # ------------ termination heuristic ------------------------------
            self.curr_infeas = self.iterate.feasibility_measure(self.iterate.x_inner_iterates, self.g_tmp, self.nlp_problem)
            self.prev_infeas = self.curr_infeas

            kappa = self.step_inf_norm/self.prev_step_inf_norm
            kappas.append(kappa)


            as_exac = cs.norm_2(
                self.direction.d_k - (self.iterate.x_inner_iterates - self.iterate.x_k)) / cs.norm_2(self.direction.d_k)
            if self.options.verbose:
                print("Kappa: ", kappa,
                      "Infeasibility", self.iterate.feasibility_measure(
                                self.iterate.x_inner_iterates, self.g_tmp, self.nlp_problem),
                      "Asymptotic Exactness: ", as_exac)

            # +1 excludes the first iteration from the kappa test
            accumulated_as_ex += as_exac
            if inner_iter % self.options.watchdog == 0:
                kappa_watch = self.step_inf_norm/watchdog_prev_inf_norm
                watchdog_prev_inf_norm = self.step_inf_norm
                if self.options.verbose:
                    print("kappa watchdog: ", kappa_watch)
                if self.curr_infeas < self.options.feasibility_tol and as_exac < 0.5:
                    self.kappa_acceptance = True
                    break
                if kappa_watch > self.options.contraction_acceptance or\
                        accumulated_as_ex/self.options.watchdog > 0.5:
                    self.kappa_acceptance = False
                    break
                accumulated_as_ex = 0

            feasibilities.append(
                self.iterate.feasibility_measure(self.iterate.x_inner_iterates, self.g_tmp, self.nlp_problem))
            asymptotic_exactness.append(as_exac)
            step_norms.append(cs.norm_inf(self.direction.d_inner_iterates))

            self.prev_step_inf_norm = self.step_inf_norm
            self.lam_tmp_g = self.lam_p_g_k
            self.lam_tmp_x = self.lam_p_x_k

        self.log.feasibility_iteration_counter += inner_iter
        self.log.inner_iters.append(inner_iter)
        self.log.inner_steps.append(step_norms)
        self.log.inner_feas.append(feasibilities)
        self.log.inner_as_exac.append(asymptotic_exactness)
        self.log.inner_kappas.append(kappas)
        self.log.asymptotic_exactness.append(as_exac)

        return self.iterate.x_inner_iterates, self.iterate.lam_g_inner_iterates, self.iterate.lam_x_inner_iterates, inner_iter
