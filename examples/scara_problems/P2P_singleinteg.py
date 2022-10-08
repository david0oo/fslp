"""
This script implements the optimization problem of a P2P motion problem
for the amazing scara robot.
This implementation describes the end effector position in the cartesian space
with a double integrator model. The start and end position are given in
cartesian space, i.e. x- and  y-coordinates are given.
The joints are calculated by inverse kinematics
"""
import casadi as cs
import numpy as np
import copy
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from py import test
import testproblems.scara_problems.scara_models.scaraParallel_workspacelimits as wspclim


def latexify():
    params = {'backend': 'ps',
              # 'text.latex.preamble': r"\usepackage{amsmath}",
              'axes.labelsize': 10,
              'axes.titlesize': 10,
              'legend.fontsize': 10,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'text.usetex': True,
              'font.family': 'serif'
              }

    matplotlib.rcParams.update(params)


latexify()


class single_integrator:

    def __init__(self, N=20):
        """
        The constructor
        """
        self.N = N

        self.n_x = 2
        self.n_u = 2
        self.n_s = 2
        self.n_sh = 3
        self.n_t = 1
        self.n_s0 = self.n_x
        self.n_sf = self.n_x

        self.ode_integrator = self.create_integrator_time_optimal()

    def create_integrator_time_optimal(self):
        """
        Returns the time optimal integrator.
        """
        x = cs.MX.sym('x', self.n_x)  
        u = cs.MX.sym('u', self.n_u)  

        xdot_temp = cs.vertcat(u[0], u[1])
        T = cs.MX.sym('T')

        ode = {'x': x, 'p': cs.vertcat(u, T), 'ode': T*xdot_temp}
        opts = {'tf': 1}
        ode_integrator = cs.integrator('ode_integrator', 'rk', ode, opts)

        return ode_integrator

    def create_problem(self, dev_lx_0=[0.0, 0.0], dev_lx_f=[0.0, 0.0], obstacleAvoidance=True):
        """
        We create the optimization problem here.
        """

        # ----- Define optimization problem -----
        self.opti = cs.Opti()

        # ----- Define optimization variables -----
        # n_var = (self.N+1)*(self.n_x) + self.N * \
        #     (self.n_u+self.n_sh) + self.n_s0 + self.n_sf + self.n_t
        n_var = (self.N+1)*(self.n_x) + self.N * \
            (self.n_u+self.n_s) + self.n_s0 + self.n_sf + self.n_t
        X_tmp = []
        U_tmp = []
        S_tmp = []
        # sh_tmp = []

        S0 = self.opti.variable(self.n_s0, 1)
        for k in range(self.N+1):
            X_tmp.append(self.opti.variable(self.n_x, 1))
            # if k == self.N+1:
            #     self.indeces_statef = list(
            #         range(self.opti.debug.x.size[1]-self.n_x, self.opti.debug.x.size[1]))
            if k < self.N:
                U_tmp.append(self.opti.variable(self.n_u, 1))
                S_tmp.append(self.opti.variable(self.n_u, 1))
                # sh_tmp.append(self.opti.variable(self.n_sh, 1))
        # Why minus 2???? because the torques do not matter, just angle at the end
        Sf = self.opti.variable(self.n_sf, 1)
        T = self.opti.variable(self.n_t, 1)

        self.indeces_S0 = list(range(self.n_s0))
        self.indeces_state0 = list(range(self.n_s0, self.n_s0+self.n_x))
        self.indeces_Sf = list(range(n_var-self.n_sf-self.n_t, n_var-self.n_t))
        self.indeces_statef = list(
            range(n_var - self.n_sf - self.n_t - self.n_x, n_var - self.n_sf  - self.n_t))

        # Transform list into variables
        X = cs.horzcat(*X_tmp)
        U = cs.horzcat(*U_tmp)
        S = cs.horzcat(*S_tmp)
        # sh = cs.horzcat(*sh_tmp)

        # ----- Define the initial state -----
        # settings = self.set_parameters()
        # N = settings['N']
        # V_max = settings['V_max']
        # self.rad = settings['safety_distance']
        # self.points = settings['obstacle_points']

        # q1_min = self.settings['q1_min']
        # q1_max = self.settings['q1_max']
        # q2_min = self.settings['q2_min']
        # q2_max = self.settings['q2_max']
        # q3_min = self.settings['q3_min']
        # q3_max = self.settings['q3_max']
        # q4_min = self.settings['q4_min']
        # q4_max = self.settings['q4_max']

        # self.q_min = cs.vertcat(q1_min, q2_min, q3_min, q4_min)
        # self.q_max = cs.vertcat(q1_max, q2_max, q3_max, q4_max)

        # Transform start and end positions from cartesian to joint space
        # Define parameters
        # iqu = [0, 2]
        # nqu = 2
        # iqv = [1, 3]
        # nqv = 2
        # q_start = self.inverse_kin_fun(
        #     settings['p_start'], cs.DM.zeros(2, 1))[0]
        # qu_start = q_start[iqu]
        # qv_start = q_start[iqv]
        # q_end = self.inverse_kin_fun(settings['p_end'], cs.DM.zeros(2, 1))[0]

        # qu_end = q_end[iqu]
        self.end_state = cs.vertcat(10,10)#cs.vertcat(settings['p_end'], cs.DM.zeros(2, 1))
        # self.end_state = cs.vertcat(qu_end, cs.DM.zeros(nqu, 1))
        # qv_end = q_end[iqv]

        # Define x0, compare with matlab line 62
        self.start_state = cs.vertcat(0, 0) #cs.vertcat(settings['p_start'], cs.DM.zeros(2, 1))
        # self.start_state = cs.vertcat(qu_start, cs.DM.zeros(nqu, 1))

        # Shooting constraints
        for k in range(self.N+1):

            if k < self.N:
                # Qm1 = U[0, k]
                # Qm3 = U[1, k]

                # Gap closing constraints
                self.opti.subject_to(self.ode_integrator(
                    x0=X[:, k], p=cs.vertcat(U[:, k], T/self.N))['xf'] == X[:, k+1])

            if k == 0:
                # Slacked Initial Condition
                self.opti.subject_to(self.start_state <= X[:, 0] + S0)
                self.opti.subject_to(X[:, 0] - S0 <= self.start_state)

            # if k > 0:
                # Path dynamic constraints
                # V2 = xee_dot**2+yee_dot**2

                # opti.subject_to(0 <= xeed)
                # opti.subject_to(0 <= yeed)

                # Limits on joint angles
                # self.opti.subject_to(self.opti.bounded(self.q_min, q, self.q_max))

                # Maximum velocity constraint
                # self.opti.subject_to(self.opti.bounded(0,  V2, V_max**2))


                # Obstacle Avoiding Constraint
                # adding a constraint that the ball must be on one side of the SH
                # self.opti.subject_to(sh[0, k-1]*xee + sh[1, k-1]
                #                      * yee + sh[2, k-1] <= -self.rad)
                # # add the constraints that the box is on the other side of the SH
                # self.opti.subject_to(sh[0, k-1]*self.points[0, 0] + sh[1, k-1]
                #                      * self.points[0, 1] + sh[2, k-1] >= 0)
                # self.opti.subject_to(sh[0, k-1]*self.points[1, 0] + sh[1, k-1]
                #                      * self.points[1, 1] + sh[2, k-1] >= 0)
                # self.opti.subject_to(sh[0, k-1]*self.points[2, 0] + sh[1, k-1]
                #                      * self.points[2, 1] + sh[2, k-1] >= 0)
                # self.opti.subject_to(sh[0, k-1]*self.points[3, 0] + sh[1, k-1]
                #                      * self.points[3, 1] + sh[2, k-1] >= 0)
                # # # Constraints on separating hyperplanes
                # self.opti.subject_to(self.opti.bounded(-1, sh[:, k-1], 1))

                # Constraints on controls
            if k < self.N:
                # Constraints on controls
                self.control_max = 10
                # infinity - norm
                # self.opti.subject_to(self.opti.bounded(-self.control_max, U[0, k], self.control_max))
                # self.opti.subject_to(self.opti.bounded(-self.control_max, U[1, k], self.control_max))
                # 2- norm
                self.opti.subject_to(U[0, k]**2 + U[1, k]**2 <= self.control_max**2)
                # 1-norm
                self.opti.subject_to(self.opti.bounded(-self.control_max, U[0, k]+U[1, k], self.control_max))
                self.opti.subject_to(self.opti.bounded(-self.control_max, U[0, k]-U[1, k], self.control_max))
                # Constraints on regularization
                self.opti.subject_to(S[0, k]==U[0, k]**2)
                self.opti.subject_to(S[1, k]==U[1, k]**2)

        # Slacked Constraints on terminal state
        self.opti.subject_to(self.end_state <= X[:, -1] + Sf)
        self.opti.subject_to(X[:, -1] - Sf <= self.end_state)
        self.opti.subject_to(self.opti.bounded(0, T, 10))

        objective = 1e5*cs.sum1(Sf) + 1e5*cs.sum1(S0) + T + 1e-5*cs.sum2(S[0,:]) + 1e-5*cs.sum2(S[1,:])

        self.opti.minimize(objective)

        # ----- Create feasible initialization -----
        p_start = self.start_state + cs.DM([0.0, 2.0])
        # q_start = self.inverse_kin_fun(p_start, cs.DM.zeros(2, 1))[0]
        # qu_start = q_start[iqu]
        # qv_start = q_start[iqv]
        # x0_init = cs.vertcat(qu_start, cs.DM.zeros(nqu, 1))
        x0_init = cs.vertcat(p_start)

        init = []

        # U_init = cs.fmin(sol.value(U) + cs.DM([4, 9]), 0) + cs.fmax(sol.value(U) - cs.DM([4, 9]), 0) #cs.fmin(sol.value(U) + 8, 0) + cs.fmax(sol.value(U) - 8, 0)
        # U_init = 0.5*sol.value(U) #cs.fmin(sol.value(U) + 8, 0) + cs.fmax(sol.value(U) - 8, 0)

        # ----- Create feasible initialization -----

        # Define initialisation for S0
        S0_plus_init = cs.fmax(self.start_state - x0_init, 0)
        S0_minus_init = cs.fmax(x0_init - self.start_state, 0)
        S0_init = cs.fmax(S0_plus_init, S0_minus_init)
        init.append(S0_init)

        # sh_init0 = cs.DM([-1, 0, 0.04])

        # u_const = cs.DM([0, 0])
        # u_const = cs.DM([1.2, -1])
        u_const = cs.DM([5, -5])#cs.DM([9, -4])

        # t_init = cs.DM([1.0])  # cs.DM([0.15])
        t_init = cs.DM([1.0])  # cs.DM([0.15])

        X_init = x0_init
        U_init = []
        S_init = []
        sh_init = []
        x_curr = x0_init
        for k in range(self.N):
            init.append(x_curr)
            x_curr = self.ode_integrator(
                x0=x_curr, p=cs.vertcat(u_const, t_init/self.N))['xf']
            X_init = cs.horzcat(X_init, x_curr)
            U_init = cs.horzcat(U_init, u_const)
            S_init = cs.horzcat(S_init, u_const**2)
            # Initialize separating hyperplanes
            # sh_init = cs.horzcat(sh_init, sh_init0)

            init.append(u_const)
            init.append(u_const**2)
            # init.append(sh_init0)

        init.append(x_curr)
        Sf_plus_init = cs.fmax(self.end_state - X_init[:, -1], 0)
        Sf_minus_init = cs.fmax(X_init[:, -1] - self.end_state, 0)
        # Sf_plus_init = cs.fmax(self.end_state - X_init[[0, 1], -1], 0)
        # Sf_minus_init = cs.fmax(X_init[[0, 1], -1] - self.end_state, 0)
        Sf_init = cs.fmax(Sf_plus_init, Sf_minus_init)
        init.append(Sf_init)
        init.append(t_init)

        self.plot_trajectory([X_init], ['init'])

        # self.opti.set_initial(sh, sh_init)
        self.opti.set_initial(X, X_init)
        self.opti.set_initial(S, S_init)
        self.opti.set_initial(Sf, Sf_init)
        self.opti.set_initial(S0, S0_init)
        self.opti.set_initial(U, U_init)
        self.opti.set_initial(T, t_init)

        self.opti.solver('ipopt', {'dump': True, 'dump_in': True, 'error_on_fail': False, 'ipopt': {
            "max_iter": 2000, 'hessian_approximation': 'exact', 'limited_memory_max_history': 5, 'print_level': 5}})
        # sol = self.opti.solve_limited()
        sol = self.opti.solve()

        # print('optimal time is: ', sol.value(T), ' seconds')
        self.plot_trajectory([X_init, sol.value(X)], ['init', 'sol'])

        x0 = cs.vertcat(*init)
        # self.check_constraints_feasibility(x0)
        x = self.opti.x
        f = self.opti.f
        g = self.opti.g

        self.n_vars = self.opti.x.shape[0]
        lbg = cs.evalf(self.opti.lbg)
        ubg = cs.evalf(self.opti.ubg)
        lbx = -cs.inf*cs.DM.ones(self.n_vars)
        ubx = cs.inf*cs.DM.ones(self.n_vars)

        return (x, f, g, lbg, ubg, lbx, ubx, x0)

    def check_constraints_feasibility(self, init):
        """
        Check for constraint violation of the initialization
        """
        settings = self.set_parameters()
        V_max = settings['V_max']

        iqu = [0, 2]
        q_end = self.inverse_kin_fun(settings['p_end'], cs.DM.zeros(2, 1))[0]
        qu_end = q_end[iqu]

        # get the states, controls, and time
        states_init = self.get_state_sol(init)
        control_init = self.get_control_sol(init)
        time_init = self.get_optimal_time(init)
        slack0 = self.get_slack0(init)
        slackf = self.get_slackf(init)
        sh_init = self.get_sh_sol(init)

        # Check for feasibility issues
        for k in range(self.N+1):
            q1 = states_init[0, k]
            q3 = states_init[1, k]
            qd1 = states_init[2, k]
            qd3 = states_init[3, k]

            # From q1 and q3 calculate q2 and q4
            qv = self.qv_analytical_fun(cs.vertcat(q1, q3))
            q2 = qv[0]
            q4 = qv[1]
            q = cs.vertcat(q1, q2, q3, q4)

            # From qd1 and qd3 retrieve
            qdv = self.depinteg_fun(q, cs.vertcat(qd1, qd3))
            qd2 = qdv[0]
            qd4 = qdv[1]
            qd = cs.vertcat(qd1, qd2, qd3, qd4)

            # Calculates the position and velocity of the 'end effector'
            (pee, _, vee, _, _, _, _) = self.end_effector_fun(
                q, qd, cs.DM.zeros(q.shape[0]))

            xee = pee[0]
            yee = pee[1]

            xee_dot = vee[0]
            yee_dot = vee[1]

            if k < self.N:
                # Qm1 = U[0, k]
                # Qm3 = U[1, k]

                # Gap closing constraints
                if not np.all((self.ode_integrator(
                    x0=states_init[:, k], p=cs.vertcat(control_init[:, k], time_init/self.N))['xf'] == states_init[:, k+1]) == True):
                    print('Shooting constraint, iteration:', k)
                

            # if k == 0:
                # Slacked Initial Condition
                # self.opti.subject_to(self.opti.bounded(0, slack0, cs.inf))
                # self.opti.subject_to(self.start_position <= states_init[:, 0] + slack0)
                # self.opti.subject_to(states_init[:, 0] - slack0 <= self.start_position)
            # add bound constraints on time variable
            # opti.subject_to(opti.bounded(0.1, T[k], 1))
            if k > 0:
                # Path dynamic constraints
                V2 = xee_dot**2+yee_dot**2

                # opti.subject_to(0 <= xeed)
                # opti.subject_to(0 <= yeed)
                # opti.subject_to(V2 <= V_max**2)

                # Limits on joint angles
                if np.any((q < self.q_min)==True) or np.any((q > self.q_max)==True):
                    print('Box constraint q, iteration:', k)

                # Maximum velocity constraint
                if np.any((V2 < 0)==True) or np.any((V2 > V_max**2)==True):
                    print('Box constraint V_max, iteration:', k)

                # Path constraints
                # opti.subject_to(X[0:5, k] <= upper_path_bounds)
                # opti.subject_to(lower_path_bounds <= X[0:5, k])

                # Obstacle Avoiding Constraint
                # adding a constraint that the ball must be on one side of the SH
                if np.any((sh_init[0, k-1]*xee + sh_init[1, k-1]
                                     * yee + sh_init[2, k-1] <= -self.rad)==False):
                    print('hyperplane constraint 1, iteration:', k)
                # add the constraints that the box is on the other side of the SH_init
                if np.any((sh_init[0, k-1]*self.points[0, 0] + sh_init[1, k-1]
                                     * self.points[0, 1] + sh_init[2, k-1] >= 0)==False):
                    print('hyperplane constraint 2, iteration:', k)
                if np.any((sh_init[0, k-1]*self.points[1, 0] + sh_init[1, k-1]
                                     * self.points[1, 1] + sh_init[2, k-1] >= 0)==False):
                    print('hyperplane constraint 3, iteration:', k)
                if np.any((sh_init[0, k-1]*self.points[2, 0] + sh_init[1, k-1]
                                     * self.points[2, 1] + sh_init[2, k-1] >= 0)==False):
                    print('hyperplane constraint 4, iteration:', k)
                if np.any((sh_init[0, k-1]*self.points[3, 0] + sh_init[1, k-1]
                                     * self.points[3, 1] + sh_init[2, k-1] >= 0)==False):
                    print('hyperplane constraint 5, iteration:', k)
                # # Constraints on separating hyperplanes
                # if np.any((self.opti.bounded(-1, sh_init[:, k-1], 1))

                # Constraints on controls
                if k < self.N:
                    # Constraints on controls
                    if np.any((control_init[0, k] < -20)==True) or np.any((control_init[0, k] > 20)==True):
                        print('Box constraint U0, iteration:', k)
                    if np.any((control_init[0, k] < -20)==True) or np.any((control_init[1, k] > 20)==True):
                        print('Box constraint U0, iteration:', k)

        # Slacked Constraints on terminal state
        # self.opti.subject_to(qu_end <= states_init[[0, 1], -1] + slackf)
        # self.opti.subject_to(states_init[[0, 1], -1] - slackf <= qu_end)
        # self.opti.subject_to(self.opti.bounded(0, slackf, cs.inf))
        # self.opti.subject_to(self.opti.bounded(1e-3, time_init, 10))
        

    def get_state_sol(self, sol):
        """
        Given the solution of the Scara robot. Get all the states.
        """
        states_sol = []
        ind_count = self.n_s0
        for k in range(self.N+1):
            states_sol.append(sol[ind_count:ind_count+self.n_x])
            ind_count += self.n_x
            if k < self.N:
                ind_count += self.n_u
                ind_count += self.n_s
                # ind_count += self.n_sh
        ind_count += 1

        return cs.horzcat(*states_sol)

    def get_control_sol(self, sol):
        """
        Given the solution of the Scara robot. Get all the constraints.
        """
        control_sol = []
        ind_count = self.n_s0
        for k in range(self.N+1):
            ind_count += self.n_x
            if k < self.N:
                control_sol.append(sol[ind_count:ind_count+self.n_u])
                ind_count += self.n_u
                ind_count += self.n_s
                # ind_count += self.n_sh
        ind_count += 1

        return cs.horzcat(*control_sol)

    def get_sh_sol(self, sol):
        """
        Given the solution of the Scara robot. Get all the constraints.
        """
        ind_count = self.n_s0
        sh_sol = []
        for k in range(self.N+1):
            ind_count += self.n_x
            if k < self.N:
                ind_count += self.n_u
                ind_count += self.n_s
                sh_sol.append(sol[ind_count:ind_count+self.n_sh])
                # ind_count += self.n_sh
        ind_count += 1

        return cs.horzcat(*sh_sol)

    def get_optimal_time(self, sol):
        """
        Extracts the optimal time from all decision variables.
        (Use after optimization) 

        Args:
            sol (Casadi DM vector): solution vector of the OCP.

        Returns:
            Casadi DM vector: The optimal time of the OCP.
        """
        time = sol[-1]
        return time

    def get_slack0(self, sol):
        """
        Extracts the slack variable at the beginning from all decision 
        variables. (Use after optimization) 

        Args:
            sol (Casadi DM vector): solution vector of the OCP.

        Returns:
            Casadi DM vector: The optimal time of the OCP.
        """
        slack0 = sol[:self.n_s0]
        return slack0

    def get_slackf(self, sol):
        """
        Extracts the slack variable at the end from all decision 
        variables. (Use after optimization) 

        Args:
            sol (Casadi DM vector): solution vector of the OCP.

        Returns:
            Casadi DM vector: The optimal time of the OCP.
        """
        slackf = sol[-self.n_sf+self.n_t:-self.n_t]
        return slackf

    def create_scaling_matrices(self, states=True, controls=True,
                                time=True, sep_hyp=False, slack0=False, slack_f=False, slacksM=False):
        """
        Creates the scaling matrices and its inverse for the feasible SLP
        solver.
        Default we set a trust-region around the states, controls, and time.
        For additional trust-regions set the bools to TRUE.
        """
        # Define the one or zero vectors, if variable is in trust-region or
        # not
        if states:
            vec_states = np.ones(self.n_x)
        else:
            vec_states = np.zeros(self.n_x)

        if controls:
            vec_controls = np.ones(self.n_u)
        else:
            vec_controls = np.zeros(self.n_u)

        if time:
            vec_time = np.ones(self.n_t)
        else:
            vec_time = np.zeros(self.n_t)

        if sep_hyp:
            vec_sep_hyp = []#np.ones(self.n_sh)
        else:
            vec_sep_hyp = []#np.zeros(self.n_sh)

        if slack0:
            vec_slack0 = np.ones(self.n_s0)
        else:
            vec_slack0 = np.zeros(self.n_s0)

        if slack_f:
            vec_slack_f = np.ones(self.n_sf)
        else:
            vec_slack_f = np.zeros(self.n_sf)

        if slacksM:
            vec_slacksM = np.ones(self.n_s)
        else:
            vec_slacksM = np.zeros(self.n_s)

        # Create the trust-region scaling matrix
        list_tr_scale_mat_vec = []
        list_tr_scale_mat_vec.append(vec_slack0)
        for k in range(self.N+1):
            list_tr_scale_mat_vec.append(vec_states)
            if k < self.N:
                list_tr_scale_mat_vec.append(vec_controls)
                list_tr_scale_mat_vec.append(vec_slacksM)
                list_tr_scale_mat_vec.append(vec_sep_hyp)
        list_tr_scale_mat_vec.append(vec_slack_f)
        list_tr_scale_mat_vec.append(vec_time)
        tr_scale_mat_vec = np.hstack(list_tr_scale_mat_vec)
        n_nonzeros = np.count_nonzero(tr_scale_mat_vec)
        row_ind = np.arange(n_nonzeros)
        col_ind = np.where(tr_scale_mat_vec == 1)[0]
        tr_scale_mat = np.zeros((n_nonzeros, self.n_vars))
        tr_scale_mat[row_ind, col_ind] = 1
        tr_scale_mat = cs.DM(tr_scale_mat)

        # Create the inverse trust-region scaling matrix
        tr_scale_mat_inv_vec = copy.copy(tr_scale_mat_vec)
        tr_scale_mat_inv_vec[tr_scale_mat_inv_vec == 0] = np.inf
        tr_scale_mat_inv = cs.diag(tr_scale_mat_inv_vec)

        return tr_scale_mat, tr_scale_mat_inv

    def plot_controls(self, sol):

        time = self.get_optimal_time(sol)
        controls = self.get_control_sol(sol)
        time_grid = np.linspace(0, np.array(time).squeeze(), self.N)
        
        plt.figure(figsize=(5,5))
        plt.subplot(211)
        plt.step(time_grid, np.array(controls[0,:].T).squeeze(), label='torqueOn1', linestyle='solid')
        plt.step(time_grid, self.control_max*np.ones(self.N), linestyle='solid')
        plt.step(time_grid, -self.control_max*np.ones(self.N), linestyle='solid')

        plt.grid(alpha=0.5)
        plt.legend(loc='upper right')

        plt.subplot(212)
        plt.step(time_grid, controls[1,:].T, label='torqueOn3', linestyle='solid')
        plt.step(time_grid, self.control_max*np.ones(self.N), linestyle='solid')
        plt.step(time_grid, -self.control_max*np.ones(self.N), linestyle='solid')
        
        plt.legend(loc='upper right')
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_states(self, sol):

        time = self.get_optimal_time(sol)
        states = self.get_state_sol(sol)
        time_grid = np.linspace(0, np.array(time).squeeze(), self.N+1)
        
        plt.figure(figsize=(5,5))
        plt.subplot(211)
        plt.step(time_grid, np.array(states[0,:].T).squeeze(), label='angle1', linestyle='solid')
        # plt.step(time_grid, self.q_min[0]*np.ones(self.N+1), linestyle='solid')
        # plt.step(time_grid, self.q_max[0]*np.ones(self.N+1), linestyle='solid')
        
        plt.grid(alpha=0.5)
        plt.legend(loc='upper right')

        plt.subplot(212)
        plt.step(time_grid, states[1,:].T, label='angle3', linestyle='solid')
        # plt.step(time_grid, self.q_min[2]*np.ones(self.N+1), linestyle='solid')
        # plt.step(time_grid, self.q_max[2]*np.ones(self.N+1), linestyle='solid')
        
        plt.legend(loc='upper right')
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_trajectory(self, list_X, list_labels):
        """
        Plot the trajectory of some given joint trajectories. 
        The given trajectories are given as angles and torques on the joints 
        by list_X.
        """
        # Transform the angles and torques into the end effector positions
        list_xee_sol = []
        list_yee_sol = []
        for i in range(len(list_X)):
            list_xee_sol.append(list_X[i][0,:])
            list_yee_sol.append(list_X[i][1,:])

        fig, ax = plt.subplots()
        # ax.set_aspect('equal')
        # rect = Rectangle(tuple(np.array(self.points[0, :]).squeeze(
        # )), 0.02, 0.02, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        for i in range(len(list_X)):
            ax.plot(list_xee_sol[i].T, list_yee_sol[i].T, label=list_labels[i])
        ax.set_xlabel('$x_1$-axis')
        ax.set_ylabel('$x_2$-axis')
        ax.set_ylim(-5, 10)
        ax.set_title('P2P-motion of singleton')
        plt.legend()
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.show()



# testproblem = single_integrator()
# testproblem.create_problem()
