"""
A logger class thats keeps track of the iterates.    
"""

class Logger:
    """
    Logger class that keeps track of interesting stats in the algorithm
    """
    def __init__(self):
        """
        Constructor
        """
        # number of function evaluations
        self.n_eval_f = 0
        # number of constraint evaluations
        self.n_eval_g = 0
        # number of gradient of objective evaluations
        self.n_eval_gradient_f = 0
        # number of Jacobian of constraints evaluations
        self.n_eval_jacobian_g = 0
        # number of Gradient of Lagrangian evaluations
        self.n_eval_gradient_lagrangian = 0
        # number of Hessian of Lagrangian evaluations
        self.n_eval_hessian_lagrangian = 0
        # number of outer iterations
        self.iteration_counter = 0
        # number of total inner iterations
        self.inner_iteration_counter = 0
        # number of accepted outer iterations
        self.accepted_iterations = 0
        # convergence status of the FSLP algorithm
        self.optimization_success = False

    def increment_n_eval_f(self):
        self.n_eval_f += 1

    def increment_n_eval_g(self):
        self.n_eval_g += 1

    def increment_n_eval_gradient_f(self):
        self.n_eval_gradient_f += 1

    def increment_n_eval_jacobian_g(self):
        self.n_eval_jacobian_g += 1

    def increment_n_eval_gradient_lagrangian(self):
        self.n_eval_gradient_lagrangian += 1

    def increment_n_eval_hessian_lagrangian(self):
        self.n_eval_hessian_lagrangian += 1

    def increment_iteration_counter(self):
        self.iteration_counter += 1

    def increment_inner_iteration_counter(self):
        self.inner_iteration_counter += 1

    def increment_inner_iteration_counter(self):
        self.inner_iteration_counter += 1
