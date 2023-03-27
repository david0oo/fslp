"""
This function contains all the necessary input given to the solver
"""
import casadi as cs

class Input:

    def __init__(self, nlp_dict: dict, initialization_dict: dict) -> None:
        
        self.x = nlp_dict['x']
        self.nx = self.x.shape[0]

        self.p = nlp_dict['p']

        # objective
        self.f = nlp_dict['f']

        # constraints
        self.g = nlp_dict['g']
        self.ng = self.g.shape[0]

        self.lam_g = cs.MX.sym('lam_g', self.ng)
        self.lam_x = cs.MX.sym('lam_x', self.nx)


        # ----------------- Refactor to different class -----------------
        if 'lbg' in initialization_dict:
            self.lbg = initialization_dict['lbg']
        else:
            self.lbg = -cs.inf*cs.DM.ones(self.ng, 1)

        if 'ubg' in initialization_dict:
            self.ubg = initialization_dict['ubg']
        else:
            self.ubg = cs.inf*cs.DM.ones(self.ng, 1)

        # Variable bounds
        if 'lbx' in initialization_dict:
            self.lbx = initialization_dict['lbx']
        else:
            self.lbx = -cs.inf*cs.DM.ones(self.nx, 1)

        if 'ubx' in initialization_dict:
            self.ubx = initialization_dict['ubx']
        else:
            self.ubx = cs.inf*cs.DM.ones(self.nx, 1)

        # Define iterative variables
        if 'x0' in initialization_dict:
            self.x0 = initialization_dict['x0']
        else:
            self.x0 = cs.DM.zeros(self.nx, 1)

        if 'lam_g0' in initialization_dict:
            self.lam_g0 = initialization_dict['lam_g0']
        else:
            self.lam_g0 = cs.DM.zeros(self.ng, 1)

        if 'lam_x0' in initialization_dict:
            self.lam_x0 = initialization_dict['lam_x0']
        else:
            self.lam_x0 = cs.DM.zeros(self.nx, 1)


        