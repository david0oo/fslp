""" 
This file contains the parameters for the Feasible Sequential Linear/Quadratic
Programming solver
"""
import casadi as cs


class Options:

    def __init__(self, nlp_dict: dict, opts: dict):
        """Initializes the parameters of the solver.

        Raises:
            KeyError: _description_
            KeyError: _description_
            KeyError: _description_
        """
        if 'tr_rad0' in opts:
            self.tr_radius0 = opts['tr_rad0']
        else:
            self.tr_radius0 = 1.0

        if 'tr_scale_mat0' in opts:
            self.tr_scale_mat0 = opts['tr_scale_mat0']
        else:
            self.tr_scale_mat0 = cs.DM.eye(nlp_dict['x'].shape[0])

        if 'tr_scale_mat_inv0' in opts:
            self.tr_scale_mat_inv0 = opts['tr_scale_mat_inv0']
        else:
            self.tr_scale_mat_inv0 = cs.inv(self.tr_scale_mat0)

        if bool(opts) and 'optimality_tol' in opts:
            self.optimality_tol = opts['optimality_tol']
        else:
            self.optimality_tol = 1e-8

        if bool(opts) and 'feasibility_tol' in opts:
            self.feasibility_tol = opts['feasibility_tol']
        else:
            self.feasibility_tol = 1e-8

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

        if bool(opts) and 'tr_radius_max' in opts:
            self.tr_radius_max = opts['tr_radius_max']
        else:
            self.tr_radius_max = 10.0

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

        if bool(opts) and 'output_level' in opts:
            self.output_level = opts['output_level']
        else:
            self.output_level = 2

        if self.verbose:
            self.output_level = 2

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

        if bool(opts) and 'subproblem_solver' in opts:
            self.subproblem_solver_name = opts['subproblem_solver']
        else:
            if self.use_sqp:
                self.subproblem_solver_name = 'qpoases'
            else:
                self.subproblem_solver_name = 'highs'

        self.subproblem_solver_opts = {}
        if bool(opts) and 'subproblem_solver_opts' in opts:
            self.subproblem_solver_opts.update(opts['subproblem_solver_opts'])
        else:
            # Needs to be nlpsol_options
            self.subproblem_solver_opts['verbose'] = True
            self.subproblem_solver_opts["print_time"] = False
            self.subproblem_solver_opts['error_on_fail'] = False

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
        # Regularization of Hessian
        if 'regularize' in opts:
            self.regularize = opts['regularize']
        else:
            self.regularize = False

        if 'regularization_factor_min' in opts:
            self.regularization_factor_min = opts['regularization_factor_min']
        else:
            self.regularization_factor_min = 1e-4

        if 'regularization_factor_max' in opts:
            self.regularization_factor_max = opts['regularization_factor_max']
        else:
            self.regularization_factor_max = 10000

        if 'regularization_factor_increase' in opts:
            self.regularization_factor_increase = opts['regularization_factor_increase']
        else:
            self.regularization_factor_increase = 10
