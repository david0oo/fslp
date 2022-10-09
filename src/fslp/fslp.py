"""
We provide a prototypical implementation of the FSLP method.
"""
import casadi as cs
import numpy as np
from timeit import default_timer as timer


cs.DM.set_precision(16)


class FSLP_Method:

    def __init__(self):
        """
        The constructor. More variables are set and initialized in specific
        functions below.
        """
        self.subproblem_solver = None
        self.solver_type = 'SLP'

    def __initialize_parameters(self, problem_dict, init_dict, opts={}):
        """
        This function initializes all parameters which are given from
        outside. This includes the problem formulation and algorithmic
        specific options.

        Args:
            problem_dict (dict): dictionary of the problem specification.
            init_dict (dict): dictionary of initialization specifications.
            opts (dict, optional): dictionary of options. Defaults to {}.
        """
        # Extract symbolic expressions from problem_dict
        # Variables
        self.x = problem_dict['x']
        self.nx = self.x.shape[0]

        # objective
        self.f = problem_dict['f']

        # constraints
        self.g = problem_dict['g']
        self.ng = self.g.shape[0]

        if 'lbg' in init_dict:
            self.lbg = init_dict['lbg']
        else:
            self.lbg = -cs.inf*cs.DM.ones(self.ng, 1)

        if 'ubg' in init_dict:
            self.ubg = init_dict['ubg']
        else:
            self.ubg = cs.inf*cs.DM.ones(self.ng, 1)

        # Variable bounds
        if 'lbx' in init_dict:
            self.lbx = init_dict['lbx']
        else:
            self.lbx = -cs.inf*cs.DM.ones(self.nx, 1)

        if 'ubx' in init_dict:
            self.ubx = init_dict['ubx']
        else:
            self.ubx = cs.inf*cs.DM.ones(self.nx, 1)

        # Define iterative variables
        if 'x0' in init_dict:
            self.x0 = init_dict['x0']
        else:
            self.x0 = cs.DM.zeros(self.nx, 1)

        if 'lam_g0' in init_dict:
            self.lam_g0 = init_dict['lam_g0']
        else:
            self.lam_g0 = cs.DM.zeros(self.ng, 1)

        if 'lam_x0' in init_dict:
            self.lam_x0 = init_dict['lam_x0']
        else:
            self.lam_x0 = cs.DM.zeros(self.nx, 1)

        if 'tr_rad0' in init_dict:
            self.tr_rad0 = init_dict['tr_rad0']
        else:
            self.tr_rad0 = 1.0

        if 'tr_scale_mat0' in init_dict:
            self.tr_scale_mat0 = init_dict['tr_scale_mat0']
        else:
            self.tr_scale_mat0 = cs.DM.eye(self.nx)

        if 'tr_scale_mat_inv0' in init_dict:
            self.tr_scale_mat_inv0 = init_dict['tr_scale_mat_inv0']
        else:
            self.tr_scale_mat_inv0 = cs.inv(self.tr_scale_mat0)

        if bool(opts) and 'optim_tol' in opts:
            self.optim_tol = opts['optim_tol']
        else:
            self.optim_tol = 1e-8

        if bool(opts) and 'feas_tol' in opts:
            self.feas_tol = opts['feas_tol']
        else:
            self.feas_tol = 1e-8

        if 'testproblem_obj' in opts:
            self.testproblem_obj = opts['testproblem_obj']
        else:
            self.testproblem_obj = None

        # Trust Region parameters
        if bool(opts) and 'tr_eta1' in opts:
            self.tr_eta1 = opts['tr_eta1']
        else:
            self.tr_eta1 = 0.25

        if bool(opts) and 'tr_eta2' in opts:
            self.tr_eta2 = opts['tr_eta2']
        else:
            self.tr_eta2 = 0.75

        if bool(opts) and 'tr_alpha1' in opts:
            self.tr_alpha1 = opts['tr_alpha1']
        else:
            self.tr_alpha1 = 0.5

        if bool(opts) and 'tr_tol' in opts:
            self.tr_tol = opts['tr_tol']
        else:
            self.tr_tol = 1e-8

        if bool(opts) and 'tr_alpha2' in opts:
            self.tr_alpha2 = opts['tr_alpha2']
        else:
            self.tr_alpha2 = 2

        if bool(opts) and 'tr_acceptance' in opts:
            self.tr_acceptance = opts['tr_acceptance']
        else:
            self.tr_acceptance = 1e-8

        if bool(opts) and 'tr_rad_max' in opts:
            self.tr_rad_max = opts['tr_rad_max']
        else:
            self.tr_rad_max = 10.0

        if bool(opts) and 'max_iter' in opts:
            self.max_iter = opts['max_iter']
        else:
            self.max_iter = 100

        if bool(opts) and 'max_inner_iter' in opts:
            self.max_inner_iter = opts['max_inner_iter']
        else:
            self.max_inner_iter = 100

        if bool(opts) and 'contraction_acceptance' in opts:
            self.contraction_acceptance = opts['contraction_acceptance']
        else:
            self.contraction_acceptance = 0.5

        if bool(opts) and 'watchdog' in opts:
            self.watchdog = opts['watchdog']
        else:
            self.watchdog = 5

        if bool(opts) and 'verbose' in opts:
            self.verbose = opts['verbose']
        else:
            self.verbose = True

        if bool(opts) and 'gradient_correction' in opts:
            self.gradient_correction = opts['gradient_correction']
        else:
            self.gradient_correction = False

        if bool(opts) and 'opt_check_slacks' in opts:
            self.opt_check_slacks = opts['opt_check_slacks']
        else:
            self.opt_check_slacks = False

        if self.opt_check_slacks:
            if bool(opts) and 'n_slacks_start' in opts:
                self.n_slacks_start = opts['n_slacks_start']
            else:
                raise KeyError('Entry n_slacks_start not specified in opts!')

            if bool(opts) and 'n_slacks_end' in opts:
                self.n_slacks_end = opts['n_slacks_end']
            else:
                raise KeyError('Entry n_slacks_end not specified in opts!')

        if bool(opts) and 'solver_type' in opts:
            if not opts['solver_type'] in ['SLP', 'SQP']:
                raise KeyError('The only allowed types are SLP or SQP!!')
            self.solver_type = opts['solver_type']
        else:
            self.solver_type = 'SLP'


        self.subproblem_sol_opts = {}
        if bool(opts) and 'subproblem_sol' in opts and opts['subproblem_sol'] != 'ipopt':
            self.subproblem_sol = opts['subproblem_sol']
        else:
            self.subproblem_sol = 'nlpsol'
            self.subproblem_sol_opts['nlpsol'] = 'ipopt'

        if bool(opts) and 'subproblem_sol_opts' in opts\
                and opts['subproblem_sol'] != 'ipopt':
            self.subproblem_sol_opts.update(opts['subproblem_sol_opts'])
        else:
            opts = {
                'ipopt': {'print_level': 0,
                          'sb': 'yes',
                          'fixed_variable_treatment': 'make_constraint',
                          'hessian_constant': 'yes',
                          'jac_c_constant': 'yes',
                          'jac_d_constant': 'yes',
                          'tol': 1e-12,
                          'tiny_step_tol': 1e-20,
                          'mumps_scaling': 0,
                          'honor_original_bounds': 'no',
                          'bound_relax_factor': 0},
                'print_time': False}
            # Needs to be nlpsol_options
            self.subproblem_sol_opts['nlpsol_options'] = opts

        if bool(opts) and 'error_on_fail' in opts:
            self.subproblem_sol_opts['error_on_fail'] = opts['error_on_fail']
        else:
            self.subproblem_sol_opts['error_on_fail'] = False

    def __initialize_functions(self, opts={}):
        """
        This function initializes the Casadi functions needed to evaluate
        the objective, constraints, gradients, jacobians, Hessians...

        Further below many functions are defined, that use the functions from
        here.
        """
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
                                                    
        if self.solver_type == 'SQP' and bool(opts) and 'hess_lag_fun' in opts:
            self.hess_lag_fun = opts['hess_lag_fun']
        else:
            self.hess_lag_fun = cs.Function('hess_lag_fun',
                                        [self.x, self.lam_g, self.lam_x],
                                        [cs.hessian(
                                            self.lagrangian,
                                            self.x)[0]])

    def __initialize_stats(self):
        """
        Initializes the statistics of the problem.
        """
        self.stats = {}
        # number of function evaluations
        self.stats['n_eval_f'] = 0
        # number of constraint evaluations
        self.stats['n_eval_g'] = 0
        # number of gradient of objective evaluations
        self.stats['n_eval_grad_f'] = 0
        # number of Jacobian of constraints evaluations
        self.stats['n_eval_jac_g'] = 0
        # number of Gradient of Lagrangian evaluations
        self.stats['n_eval_grad_lag'] = 0
        # number of Hessian of Lagrangian evaluations
        self.stats['n_eval_hess_l'] = 0
        # number of outer iterations
        self.stats['iter_count'] = 0
        # number of total inner iterations
        self.stats['inner_iter'] = 0
        # number of accepted outer iterations
        self.stats['accepted_iter'] = 0
        # convergence status of the FSLP algorithm
        self.stats['success'] = False

    def feasibility_measure(self, x, g_x):
        """
        The feasibility measure in the l-\\infty norm.

        Args:
            x (DM-array): value of state variable
            g_x (DM-array): value of constraints at state variable

        Returns:
            double: the feasibility in the l-\\infty norm
        """
        return np.array(cs.norm_inf(cs.vertcat(
                        cs.fmax(0, self.lbg-g_x),
                        cs.fmax(0, g_x-self.ubg),
                        cs.fmax(0, self.lbx-x),
                        cs.fmax(0, x-self.ubx)))).squeeze()

    def __create_subproblem_solver(self):
        """
        This function creates an LP-solver object with the casadi conic 
        operation.
        """
        if self.solver_type == 'SQP':
            B_placeholder = self.hess_lag_fun(self.x0, self.lam_g0, self.lam_x0)

            qp_struct = {   'h': B_placeholder.sparsity(),
                            'a': self.A_k.sparsity()}
            self.subproblem_solver = cs.conic(  "qp_solver",
                                                self.subproblem_sol,
                                                qp_struct,
                                                self.subproblem_sol_opts)
        else:
            lp_struct = {'a': self.A_k.sparsity()}

            self.subproblem_solver = cs.conic(  "lp_solver",
                                                self.subproblem_sol,
                                                lp_struct,
                                                self.subproblem_sol_opts)

    def solve_lp(self,
                 g=None,
                 a=None,
                 lba=None,
                 uba=None,
                 lbx=None,
                 ubx=None):
        """
        This function solves the lp subproblem. Additionally some processing
        of the result is done and the statistics are saved. The input signature
        is the same as for a casadi lp solver.

        Args:
            g (Casadi DM vector, optional): Objective vector. Defaults to None.
            a (Casadi DM array, optional): Constraint matrix. Defaults to None.
            lba (Casadi DM vector, optional): Lower bounds on constraint 
                                              matrix. Defaults to None.
            uba (Casadi DM vector, optional): Upper bounds on constraint
                                              matrix. Defaults to None.
            lbx (Casadi DM vector, optional): Lower bounds on states. 
                                              Defaults to None.
            ubx (Casadi DM vector, optional): Upper bounds on states.
                                              Defaults to None.

        Returns:
            (bool, Casadi DM vector, Casadi DM vector): First indicates of LP
            was solved succesfully, second entry is the new search direction, 
            the third entry are the lagrange multipliers for the new
            search direction
        """
        res = self.subproblem_solver(   g=g,
                                        a=a,
                                        lba=lba,
                                        uba=uba,
                                        lbx=lbx,
                                        ubx=ubx)

        # Keep track that bounds of QP are guaranteed. If not because of a
        # tolerance, make them exact.
        p_tmp = res['x']
        # Get indeces where variables are violated
        lower_p = list(np.nonzero(np.array(p_tmp < lbx).squeeze())[0])
        upper_p = list(np.nonzero(np.array(p_tmp > ubx).squeeze())[0])

        # Resolve the 'violation' in the search direction
        if bool(lower_p):
            p_tmp[lower_p] = lbx[lower_p]
        if bool(upper_p):
            p_tmp[upper_p] = ubx[upper_p]

        # Process the new search directions and multipliers w/o slacks
        p = p_tmp
        lam_p_g = res['lam_a']
        lam_p_x = res['lam_x']

        solve_success = self.subproblem_solver.stats()['success']

        return (solve_success, p, lam_p_g, lam_p_x)

    def solve_qp(self,
                 h=None,
                 g=None,
                 a=None,
                 lba=None,
                 uba=None,
                 lbx=None,
                 ubx=None):
        """
        This function solves the qp subproblem. Additionally some processing of
        the result is done and the statistics are saved. The input signature is
        the same as for a casadi qp solver.
        Input:
        h       Matrix in QP objective
        g       Vector in QP objective
        a       Matrix for QP constraints
        lba     lower bounds of constraints
        uba     upper bounds of constraints
        lbx     lower bounds of variables
        ubx     upper bounds of variables

        Return:
        solve_success   Bool, indicating if qp was succesfully solved
        p_scale         Casadi DM vector, the new search direction
        lam_p_scale     Casadi DM vector, the lagrange multipliers for the new
                        search direction
        """
        res = self.subproblem_solver(   h=h,
                                        g=g,
                                        a=a,
                                        lba=lba,
                                        uba=uba,
                                        lbx=lbx,
                                        ubx=ubx)
        
        # Save some statistics

        # Keep track that bounds of QP are guaranteed. If not because of a 
        # tolerance, make them exact.

        p_tmp = res['x']
        # Get indeces where variables are violated
        lower_p = list(np.nonzero(np.array(p_tmp < lbx).squeeze())[0])
        upper_p = list(np.nonzero(np.array(p_tmp > ubx).squeeze())[0])

        # Resolve the 'violation' in the search direction
        if bool(lower_p):
            p_tmp[lower_p] = lbx[lower_p]
        if bool(upper_p):
            p_tmp[upper_p] = ubx[upper_p]

        # Process the new search directions and multipliers w/o slacks
        p = p_tmp
        lam_p_g = res['lam_a']
        lam_p_x = res['lam_x']
        
        solve_success = self.subproblem_solver.stats()['success']

        return (solve_success, p, lam_p_g, lam_p_x)

    def solve_subproblem(   self,
                            g=None,
                            lba=None,
                            uba=None,
                            lbx=None,
                            ubx=None):
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
        if self.solver_type == 'SQP':
            return self.solve_qp(   h=self.H_k,
                                    g=g,
                                    a=self.A_k,
                                    lba=lba,
                                    uba=uba,
                                    lbx=lbx,
                                    ubx=ubx)
        else:
            return self.solve_lp(   g=g,
                                    a=self.A_k,
                                    lba=lba,
                                    uba=uba,
                                    lbx=lbx,
                                    ubx=ubx)
        

    def __set_optimal_slack_step(self, x, p):
        """
        Sets the slack variables at the current iterate to the optimal slack 
        value w.r.t. the current start and end state. Then the slack variables
        should have the minimal value.

        Args:
            x (Casadi DM vector): the value of the current iterate 
            p (Casadi DM vector): the new search direction

        Returns:
            Casadi DM vector: the adjusted search direction with minimal slacks
        """
        if self.testproblem_obj is not None:
            start_state = self.testproblem_obj.start_state
            end_state = self.testproblem_obj.end_state
            x_new = x + p

            curr_state_0 = x_new[self.testproblem_obj.indeces_state0]
            curr_state_f = x_new[self.testproblem_obj.indeces_statef]

            S0_plus = cs.fmax(start_state - curr_state_0, 0)
            S0_minus = cs.fmax(curr_state_0 - start_state, 0)
            S0_init = cs.fmax(S0_plus, S0_minus)

            Sf_plus_init = cs.fmax(end_state - curr_state_f, 0)
            Sf_minus_init = cs.fmax(curr_state_f - end_state, 0)
            Sf_init = cs.fmax(Sf_plus_init, Sf_minus_init)

            x_new[self.testproblem_obj.indeces_S0] = S0_init
            x_new[self.testproblem_obj.indeces_Sf] = Sf_init

            p_new = x_new - x
        else:
            p_new = p

        return p_new

    def __eval_grad_jac(self, step_accepted=False):
        """ 
        Evaluate functions, gradient, jacobian at current iterate x_k.

        Args:
            step_accepted (bool, optional): Denotes if previous step was
            accepted. In an accepted step the gradient of the constraints do
            not need to be re-evaluated. Defaults to False.
        """
        self.val_f_k = self.__eval_f(self.x_k)
        if step_accepted:
            self.val_g_k = self.g_tmp
        else:
            self.val_g_k = self.__eval_g(self.x_k)
        self.val_grad_f_k = self.__eval_grad_f(self.x_k)
        self.val_jac_g_k = self.__eval_jac_g(self.x_k)
        if self.solver_type == 'SQP':
            self.hess_lag_k = self.__eval_hess_l(self.x_k,
                                                 self.lam_g_k,
                                                 self.lam_x_k)

    def __eval_f(self, x):
        """
        Evaluates the objective function. And stores the statistics of it.

        Args:
            x (Casadi DM vector): the value of the states where f is evaluated

        Returns:
            Casadi DM scalar: the value of f at the given x.
        """
        self.stats['n_eval_f'] += 1
        return self.f_fun(x)

    def __eval_g(self, x):
        """
        Evaluates the constraint function. And stores the statistics of it.

        Args:
            x (Casadi DM vector): the value of the states where g is evaluated

        Returns:
            _type_: _description_
        """
        self.stats['n_eval_g'] += 1
        return self.g_fun(x)

    def __eval_grad_f(self, x):
        """
        Evaluates the objective gradient function. And stores the statistics 
        of it.
        
        Args:
            x (Casadi DM vector): the value of the states where gradient of f 
            is evaluated

        Returns:
           Casadi DM vector: the value of g at the given x.
        """
        self.stats['n_eval_grad_f'] += 1
        return self.grad_f_fun(x)

    def __eval_jac_g(self, x):
        """
        Evaluates the constraint jacobian function. And stores the
        statistics of it.

        Args:
            x (Casadi DM vector): the value of the states where Jacobian of g 
            is evaluated

        Returns:
            Casadi DM vector: the value of g at the given x.
        """
        self.stats['n_eval_jac_g'] += 1
        return self.jac_g_fun(x)

    def __eval_grad_lag(self, x, lam_g, lam_x):
        """
        Evaluates the gradient of the Lagrangian at x, lam_g, and lam_x.
        
        Args:
            x (Casadi DM vector): the value of the states where Jacobian of g 
            lam_g (Casadi DM vector): value of multipliers for constraints g
            lam_x (Casadi DM vector): value of multipliers for states x

        Returns:
            Casadi DM vector: the value of gradient of Lagrangian
        """
        self.stats['n_eval_grad_lag'] += 1
        return self.grad_lag_fun(x, lam_g, lam_x)

    def __eval_hess_l(self, x, lam_g, lam_x):
        """
        Evaluates the Hessian of Lagrangian. And stores the statistics 
        of it.
        """
        self.stats['n_eval_hess_l'] += 1
        return self.hess_lag_fun(x, lam_g, lam_x)

    def __prepare_subproblem_matrices(self):
        """
        Prepares the objective vector g and the constraint matrix A for the LP.
        I.e., min_{x}   g_k.T x
              s.t.  lba <= A_k*x <= uba,
                    lbx <= x <= ubx.
        """
        self.A_k = self.val_jac_g_k
        self.g_k = self.val_grad_f_k
        if self.solver_type == 'SQP':
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

    def __check_slacks_zero(self):
        """
        This function checks if the slack variables are zero.
        """
        if self.opt_check_slacks:
            s0 = self.x_k[0:self.n_slacks_start]
            sf = self.x_k[-(1+self.n_slacks_end):-1]
            s = cs.vertcat(s0, sf)
            if not self.slacks_zero and cs.norm_inf(s) < self.feas_tol:
                self.slacks_zero_t_wall = timer() - self.lasttime
                self.slacks_zero = True
                self.slacks_zero_iter = self.n_iter
                self.slacks_zero_n_eval_g = self.stats['n_eval_g']

    def feasibility_iterations(self, p):
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
        self.x_tmp = self.x_k + p_tmp
        self.g_tmp = self.__eval_g(self.x_tmp)

        self.kappa_acceptance = False

        inner_iter = 0
        asymptotic_exactness = []
        as_exac = cs.norm_2(
                self.p_k - (self.x_tmp - self.x_k)) / cs.norm_2(self.p_k)
        self.prev_infeas = self.feasibility_measure(self.x_tmp, self.g_tmp)
        self.curr_infeas = self.feasibility_measure(self.x_tmp, self.g_tmp)
        feasibilities = [self.prev_infeas]
        step_norms = []
        kappas = []

        watchdog_prev_inf_norm = self.prev_step_inf_norm
        accumulated_as_ex = 0

        for j in range(self.max_inner_iter):

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

            # 'Relative' version TR Methods book
            # d = self.g_tmp
            # lba_correction = self.lbg - d
            # uba_correction = self.ubg - d

            if self.gradient_correction:
                grad_L_tmp = self.__eval_grad_lag(self.x_tmp, lam_p_g_tmp, lam_p_x_tmp)
                print('Gradient of Lagrangian: ', cs.norm_inf(grad_L_tmp))
                # Do the gradient correction, could also be + instead of -??
                grad_f_correction = grad_L_tmp - self.A_k.T @ lam_p_g_tmp - lam_p_x_tmp#self.tr_scale_mat_k.T @ lam_p_x_tmp
            else:
                # Do just Zero-Order Iterations
                if self.solver_type == 'SQP':
                    grad_f_correction = self.val_grad_f_k + self.H_k @ (self.x_tmp - self.x_k)
                else:
                    grad_f_correction = self.val_grad_f_k


            lbp = cs.fmax(-self.tr_rad_k*self.tr_scale_mat_inv_k @
                          cs.DM.ones(self.nx, 1) - (self.x_tmp-self.x_k),
                          self.lbx - self.x_tmp)
            ubp = cs.fmin(self.tr_rad_k*self.tr_scale_mat_inv_k @
                          cs.DM.ones(self.nx, 1) - (self.x_tmp-self.x_k),
                          self.ubx - self.x_tmp)

            lba_correction = self.lbg - self.g_tmp
            uba_correction = self.ubg - self.g_tmp
            lb_var_correction = lbp
            ub_var_correction = ubp

            (_,
            p_tmp,
            lam_p_g_tmp,
            lam_p_x_tmp) = self.solve_subproblem(   g=grad_f_correction,
                                                    lba=lba_correction,
                                                    uba=uba_correction,
                                                    lbx=lb_var_correction,
                                                    ubx=ub_var_correction)

            p_tmp = self.__set_optimal_slack_step(self.x_tmp, p_tmp)

            self.step_inf_norm = cs.norm_inf(self.tr_scale_mat_k @ p_tmp)

            # Old version of step update. Beware that here multiplier term is always zero because of wrong update!
            # self.step_inf_norm = cs.fmax(cs.norm_inf(p_tmp),
            #                              cs.fmax(
            #                                  cs.norm_inf(
            #                                      self.lam_tmp_g-self.lam_p_g_k),
            #                                  cs.norm_inf(self.lam_tmp_x-
            #                                              self.lam_p_x_k)))

            self.x_tmp = self.x_tmp + p_tmp
            self.g_tmp = self.__eval_g(self.x_tmp)  # x_tmp = x_{tmp-1} + p_tmp

            self.curr_infeas = self.feasibility_measure(self.x_tmp, self.g_tmp)
            self.prev_infeas = self.curr_infeas

            kappa = self.step_inf_norm/self.prev_step_inf_norm
            kappas.append(kappa)
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

    def eval_m_k(self, p):
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
        if self.solver_type == 'SQP':
            return self.val_grad_f_k.T @ p + 0.5 * p.T @ self.H_k @ p
        else:
            return self.val_grad_f_k.T @ p

    def eval_trust_region_ratio(self):
        """
        We evaluate the trust region ratio here.

        rho = (f(x_k) - f(x_k + p_k_correction)) / (-m_k(p_k))

        x_k is the current iterate
        p_k is the solution of the original QP
        p_k_correction comes from the feasibility iterations
        """
        f_correction = self.f_fun(self.x_k_correction)
        self.list_mks.append(cs.fabs(self.m_k))
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

    def __complementarity_condition(self):
        """
        Calculate inf-norm of the complementarity condition.
        """

        pre_compl_g = cs.fmin(cs.fabs(self.lbg - self.val_g_k),
                              cs.fabs(self.val_g_k - self.ubg))
        # is this sufficient?? because if there is no constraint we should not
        # care about it
        pre_compl_g[list(np.nonzero(pre_compl_g == cs.inf)[0])] = 0

        compl_g = cs.mmax(pre_compl_g * self.lam_g_k)

        pre_compl_x = cs.fmin(cs.fabs(self.lbx - self.x_k),
                              cs.fabs(self.x_k - self.ubx))

        # same here. Is this sufficient??
        pre_compl_x[list(np.nonzero(pre_compl_x == cs.inf)[0])] = 0

        compl_x = cs.mmax(pre_compl_x * self.lam_x_k)

        return cs.fmax(compl_g, compl_x)

    def print_output(self, i):
        """
        This function prints the iteration output to the console.

        Args:
            i (integer): the iteration index of the solve operator.
        """
        if i % 10:
            print(("{iter:>6} | {m_k:^10} | {grad_lag:^10} | {feas:^10} | "
                   "{compl:^10} | {f_x:^10} | {lam_g:^13} | {lam_x:^13} | "
                   "{feas_it:^9} | {tr_rad:^13} | {tr_ratio:^10}").format(
                        iter='iter', m_k='m_k', grad_lag='grad_lag',
                        compl='compl', f_x='f(x)', lam_g='||lamg||_inf',
                        lam_x='||lamx||_inf', feas='infeas',
                        feas_it='feas iter', tr_rad='tr_rad',
                        tr_ratio='tr_ratio'))
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

    def solve(self, problem_dict, init_dict, opts={}, callback=None):
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
        if not bool(init_dict):  # check whether empty
            raise ValueError("Error: You should specify an init_dict!")

        if not bool(problem_dict):  # check whether empty
            raise ValueError("Error: You should specify a problem_dict!")
        self.lasttime = timer()

        self.__initialize_parameters(problem_dict, init_dict, opts)
        self.__initialize_functions(opts)
        self.__initialize_stats()

        self.x_k = self.x0
        self.lam_g_k = self.lam_g0
        self.lam_x_k = self.lam_x0
        self.__eval_grad_jac()

        if self.feasibility_measure(self.x_k, self.val_g_k) > self.feas_tol:
            raise ValueError('Initial guess needs to be feasible!!')

        self.slacks_zero = False
        self.slacks_zero_iter = 0
        self.slacks_zero_n_eval_g = 0

        self.step_accepted = False
        self.tr_scale_mat_inv_k = self.tr_scale_mat_inv0
        self.tr_scale_mat_k = self.tr_scale_mat0
        self.rho_k = 0

        self.success = False

        self.lam_tmp_g = self.lam_g0
        self.lam_tmp_x = self.lam_x0

        self.inner_iter_count = 0
        self.inner_iter_limit_count = 0

        self.tr_rad_k = self.tr_rad0
        self.__prepare_subproblem_matrices()
        self.__prepare_subproblem_bounds_variables()
        self.__prepare_subproblem_bounds_constraints()

        self.__create_subproblem_solver()

        self.feas_iter = -1
        self.m_k = -1

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

        for i in range(self.max_iter):
            # Do the FSLP outer iterations here
            # print iteration here
            if self.verbose:
                self.print_output(i)

            self.n_iter = i + 1
            self.lam_tmp_g = self.lam_g_k
            self.lam_tmp_x = self.lam_x_k

            self.tr_radii.append(self.tr_rad_k)

            
            (solve_success,
            self.p_k,
            self.lam_p_g_k,
            self.lam_p_x_k) = self.solve_subproblem(    g=self.g_k,
                                                        lba=self.lba_k,
                                                        uba=self.uba_k,
                                                        lbx=self.lb_var_k,
                                                        ubx=self.ub_var_k)

            
            if not solve_success:
                if self.solver_type == 'SQP':
                    # TODO: correct this
                    print('Something went wrong in QP: ', self.subproblem_solver.stats()[
                        'return_status'])
                else:
                    print('Something went wrong in LP: ', self.subproblem_solver.stats()[
                        'return_status'])
                break

            self.p_k = self.__set_optimal_slack_step(self.x_k, self.p_k)
            self.m_k = self.eval_m_k(self.p_k)

            if cs.fabs(self.m_k) < self.optim_tol:
                if self.verbose:
                    print('Optimal Point Found? Linear model is zero.')
                self.success = True
                self.stats['success'] = True
                break

            self.step_inf_norm = cs.norm_inf(self.tr_scale_mat_k @ self.p_k)

            # Old version of step inf norm!
            # self.step_inf_norm = cs.fmax(
            #     cs.norm_inf(self.p_k),
            #     cs.fmax(
            #         cs.norm_inf(self.lam_tmp_g-self.lam_p_g_k),
            #         cs.norm_inf(self.lam_tmp_x-self.lam_p_x_k)
            #     )
            # )

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
