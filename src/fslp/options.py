""" 
This file contains the parameters for the Feasible Sequential Linear/Quadratic
Programming solver
"""
import casadi as cs


class Options:

    def __init__(self, opts: dict):
        """Initializes the parameters of the solver.

        Raises:
            KeyError: _description_
            KeyError: _description_
            KeyError: _description_
        """
        if 'tr_rad0' in opts:
            self.tr_rad0 = opts['tr_rad0']
        else:
            self.tr_rad0 = 1.0

        if 'tr_scale_mat0' in opts:
            self.tr_scale_mat0 = opts['tr_scale_mat0']
        else:
            self.tr_scale_mat0 = cs.DM.eye(self.nx)

        if 'tr_scale_mat_inv0' in opts:
            self.tr_scale_mat_inv0 = opts['tr_scale_mat_inv0']
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


        if bool(opts) and 'solver_type' in opts:
            if not opts['solver_type'] in ['SLP', 'SQP']:
                raise KeyError('The only allowed types are SLP or SQP!!')
            self.solver_type = opts['solver_type']
        else:
            self.solver_type = 'SLP'
        if bool(opts) and 'use_anderson' in opts:
            self.use_anderson = opts['use_anderson']
        else:
            self.use_anderson = False

        if bool(opts) and 'anderson_memory' in opts:
            self.sz_anderson_memory = opts['anderson_memory']
        else:
            self.sz_anderson_memory = 1

        if self.solver_type == 'SQP':
            print("SOLVING PROBLEM WITH SQP!\n")
            self.use_sqp = True
        else:
            print("SOLVING PROBLEM WITH SLP!\n")
            self.use_sqp = False

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
            self.subproblem_sol_opts['verbose'] = True
            self.subproblem_sol_opts["print_time"] = False


        if bool(opts) and 'error_on_fail' in opts:
            self.subproblem_sol_opts['error_on_fail'] = opts['error_on_fail']
        else:
            self.subproblem_sol_opts['error_on_fail'] = False

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

        if 'testproblem_obj' in opts:
            self.testproblem_obj = opts['testproblem_obj']
        else:
            self.testproblem_obj = None
        # -----------------------------------------------------------