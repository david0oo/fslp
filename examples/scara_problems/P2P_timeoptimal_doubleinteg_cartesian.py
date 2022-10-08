"""
This script implements the optimization problem of a P2P motion problem
for the amazing scara robot. 
The start and end position are given in cartesian space, i.e. x- and 
y-coordinates are given. Then, we transform it into joint space and 
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


class scara_problem:

    def __init__(self, N=20):
        """
        The constructor
        """
        self.N = N

        # Define dynamics of system
        self.dynamics = cs.Function.load(
            'scara_models/scaraParallel_cartesian_doubleinteg.casadi')

        self.n_x = self.dynamics.size1_in(0)
        self.n_z = self.dynamics.size1_in(1)
        self.n_u = self.dynamics.size1_in(2)
        self.n_sh = 3
        self.n_t = 1
        self.n_s0 = self.n_x
        self.n_sf = self.n_x - 2  # why?????

        # load casadi functions for cartesian to joint conversions and vice versa
        self.inverse_kin_fun = cs.Function.load(
            'scara_models/scaraParallel_inverse_kinematic.casadi')
        self.end_effector_fun = cs.Function.load(
            'scara_models/scaraParallel_sensor_end_effector.casadi')
        self.depinteg_fun = cs.Function.load(
            'scara_models/scaraParallel_depinteg.casadi')
        self.qv_analytical_fun = cs.Function.load(
            'scara_models/scaraParallel_qv_analytical.casadi')

        self.ode_integrator = self.create_integrator_time_optimal()

    def create_integrator_time_optimal(self):
        """
        Returns the time optimal integrator.
        """
        x = cs.MX.sym('z', self.n_x)  # cs.vertcat(q1, q3, qd1, qd3)
        z = cs.MX.sym('z', self.n_z)
        u = cs.MX.sym('z', self.n_u)  # cs.vertcat(Qm1, Qm3)

        xdot_temp = self.dynamics(x, [], u)[0]
        T = cs.MX.sym('T')

        ode = {'x': x, 'p': cs.vertcat(u, T), 'ode': T*xdot_temp}
        opts = {'tf': 1}
        ode_integrator = cs.integrator('ode_integrator', 'rk', ode, opts)

        return ode_integrator

    def set_parameters(self):
        self.settings = {}
        self.settings['T'] = 0.25  # Time horizon
        self.settings['N'] = self.N  # number of time interval

        # P2P related
        d = 0.0
        x_offset = 0.0
        # start and end point carthesian coordinate
        # starting point based on ID2CON validation
        self.settings['p_start'] = cs.vertcat(0.0+x_offset, 0.115)
        # end point based on ID2CON validation
        self.settings['p_end'] = cs.vertcat(0.0+x_offset,  0.405)
        # obstacles coordinates
        cog = cs.vertcat(x_offset, 0.2)
        c = 0.02
        self.settings['obstacle_points'] = cs.vertcat(cs.horzcat(cog[0]-c/2, cog[1]-c/2),
                                                      cs.horzcat(
            cog[0]+c/2, cog[1]-c/2),
            cs.horzcat(
            cog[0]+c/2, cog[1]+c/2),
            cs.horzcat(cog[0]-c/2, cog[1]+c/2))
        self.settings['weights_value'] = cs.vertcat(1e2, 1e-1)

        # maximum speed in carthesian space --> NEED to be replaced by actuator
        # limitations in joint space
        self.settings['V_max'] = 2.0
        self.settings['safety_distance'] = 0.001  # used as hard constraints!

        # Limitations of joint angles
        self.settings['q1_min'] = -cs.pi/6
        self.settings['q1_max'] = 5*cs.pi/6
        self.settings['q2_min'] = -11*cs.pi/12
        self.settings['q2_max'] = 11*cs.pi/12
        self.settings['q3_min'] = cs.pi - 5*cs.pi/6
        self.settings['q3_max'] = cs.pi - (-cs.pi/6)
        self.settings['q4_min'] = -11*cs.pi/12
        self.settings['q4_max'] = 11*cs.pi/12

        return self.settings

    def create_problem(self, dev_lx_0=[0.0, 0.0], dev_lx_f=[0.0, 0.0]):
        """
        We create the optimization problem here.
        """

        # ----- Define optimization problem -----
        self.opti = cs.Opti()

        # ----- Define optimization variables -----
        n_var = (self.N+1)*(self.n_x) + self.N * \
            (self.n_u+self.n_sh) + self.n_s0 + self.n_sf + self.n_t
        X_tmp = []
        U_tmp = []
        # sh_tmp = []

        S0 = self.opti.variable(self.n_s0, 1)
        for k in range(self.N+1):
            X_tmp.append(self.opti.variable(self.n_x, 1))
            if k == self.N+1:
                self.indeces_statef = list(
                    range(self.opti.debug.x.size[1]-self.n_x, self.opti.debug.x.size[1]))
            if k < self.N:
                U_tmp.append(self.opti.variable(self.n_u, 1))
                # sh_tmp.append(self.opti.variable(self.n_sh, 1))
        # Why minus 2???? because the torques do not matter, just angle at the end
        Sf = self.opti.variable(self.n_sf, 1)
        T = self.opti.variable(self.n_t, 1)

        self.indeces_S0 = list(range(self.n_s0))
        self.indeces_state0 = list(range(self.n_s0, self.n_s0+self.n_x))
        self.indeces_Sf = list(range(n_var-self.n_sf-self.n_t, n_var-self.n_t))
        self.indeces_statef = list(
            range(n_var - self.n_sf - self.n_t - self.n_x, n_var - self.n_sf - self.n_t))

        # Transform list into variables
        X = cs.horzcat(*X_tmp)
        U = cs.horzcat(*U_tmp)
        # sh = cs.horzcat(*sh_tmp)

        # ----- Define the initial state -----
        settings = self.set_parameters()
        N = settings['N']
        V_max = settings['V_max']
        self.rad = settings['safety_distance']
        self.points = settings['obstacle_points']

        q1_min = self.settings['q1_min']
        q1_max = self.settings['q1_max']
        q2_min = self.settings['q2_min']
        q2_max = self.settings['q2_max']
        q3_min = self.settings['q3_min']
        q3_max = self.settings['q3_max']
        q4_min = self.settings['q4_min']
        q4_max = self.settings['q4_max']

        q_min = cs.vertcat(q1_min, q2_min, q3_min, q4_min)
        q_max = cs.vertcat(q1_max, q2_max, q3_max, q4_max)

        # Transform start and end positions from cartesian to joint space
        # Define parameters
        iqu = [0, 2]
        nqu = 2
        iqv = [1, 3]
        nqv = 2
        q_start = self.inverse_kin_fun(
            settings['p_start'], cs.DM.zeros(2, 1))[0]
        qu_start = q_start[iqu]
        qv_start = q_start[iqv]
        q_end = self.inverse_kin_fun(settings['p_end'], cs.DM.zeros(2, 1))[0]
        qu_end = q_end[iqu]
        qv_end = q_end[iqv]

        # Define x0, compare with matlab line 62
        self.start_position = cs.vertcat(qu_start, cs.DM.zeros(nqu, 1))

        # Shooting constraints
        for k in range(self.N+1):
            q1 = X[0, k]
            q3 = X[1, k]
            qd1 = X[2, k]
            qd3 = X[3, k]

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
                self.opti.subject_to(self.ode_integrator(
                    x0=X[:, k], p=cs.vertcat(U[:, k], T/self.N))['xf'] == X[:, k+1])

            if k == 0:
                # Slacked Initial Condition
                self.opti.subject_to(self.opti.bounded(0, S0, cs.inf))
                self.opti.subject_to(self.start_position <= X[:, 0] + S0)
                self.opti.subject_to(X[:, 0] - S0 <= self.start_position)
            # add bound constraints on time variable
            # opti.subject_to(opti.bounded(0.1, T[k], 1))
            if k > 0:
                # Path dynamic constraints
                V2 = xee_dot**2+yee_dot**2

                # opti.subject_to(0 <= xeed)
                # opti.subject_to(0 <= yeed)
                # opti.subject_to(V2 <= V_max**2)

                # Limits on joint angles
                self.opti.subject_to(self.opti.bounded(q_min, q, q_max))

                # Maximum velocity constraint
                self.opti.subject_to(self.opti.bounded(1e-3,  V2, V_max**2))

                # Path constraints
                # opti.subject_to(X[0:5, k] <= upper_path_bounds)
                # opti.subject_to(lower_path_bounds <= X[0:5, k])

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
                # self.opti.subject_t o(sh[0, k-1]*self.points[3, 0] + sh[1, k-1]
                #                      * self.points[3, 1] + sh[2, k-1] >= 0)
                # # Constraints on separating hyperplanes
                # self.opti.subject_to(self.opti.bounded(-1, sh[:, k-1], 1))

                # Constraints on controls
                if k < self.N:
                    # Constraints on controls
                    self.opti.subject_to(self.opti.bounded(-20, U[0, k], 20))
                    self.opti.subject_to(self.opti.bounded(-20, U[1, k], 20))

        # Slacked Constraints on terminal state
        self.opti.subject_to(qu_end <= X[[0, 1], -1] + Sf)
        self.opti.subject_to(X[[0, 1], -1] - Sf <= qu_end)
        self.opti.subject_to(self.opti.bounded(0, Sf, cs.inf))
        self.opti.subject_to(self.opti.bounded(0.5, T, 10))

        objective = 1e5*cs.sum1(Sf) + 1e5*cs.sum1(S0) + T

        self.opti.minimize(objective)

        # ----- Create feasible initialization -----
        p_start = settings['p_start'] + cs.DM([0.05, 0.05])
        q_start = self.inverse_kin_fun(p_start, cs.DM.zeros(2, 1))[0]
        qu_start = q_start[iqu]
        qv_start = q_start[iqv]
        x0_init = cs.vertcat(qu_start, cs.DM.zeros(nqu, 1))

        init = []

        # U_init = cs.fmin(sol.value(U) + cs.DM([4, 9]), 0) + cs.fmax(sol.value(U) - cs.DM([4, 9]), 0) #cs.fmin(sol.value(U) + 8, 0) + cs.fmax(sol.value(U) - 8, 0)
        # U_init = 0.5*sol.value(U) #cs.fmin(sol.value(U) + 8, 0) + cs.fmax(sol.value(U) - 8, 0)

        # Define initialisation for S0
        S0_plus_init = cs.fmax(self.start_position - x0_init, 0)
        S0_minus_init = cs.fmax(x0_init - self.start_position, 0)
        S0_init = cs.fmax(S0_plus_init, S0_minus_init)
        init.append(S0_init)

        sh_init0 = cs.DM([-1, 0, 0.05])

        # u_const = cs.DM([0, 0])
        # u_const = cs.DM([1.2, -1])
        u_const = cs.DM([0.07, -0.055])

        t_init = cs.DM([1.0])  # cs.DM([0.15])

        X_init = x0_init
        U_init = []
        sh_init = []
        x_curr = x0_init
        for k in range(N):
            init.append(x_curr)
            x_curr = self.ode_integrator(
                x0=x_curr, p=cs.vertcat(u_const, t_init/N))['xf']
            # x_curr = self.ode_integrator(x0=x_curr, p=cs.vertcat(U_init[:,k], t_init/N))['xf']
            X_init = cs.horzcat(X_init, x_curr)
            U_init = cs.horzcat(U_init, u_const)
            # Initialize separating hyperplanes
            # sh_init = cs.horzcat(sh_init, sh_init0)

            init.append(u_const)
            # init.append(sh_init0)

        init.append(x_curr)
        Sf_plus_init = cs.fmax(qu_end - X_init[[0, 1], -1], 0)
        Sf_minus_init = cs.fmax(X_init[[0, 1], -1] - qu_end, 0)
        Sf_init = cs.fmax(Sf_plus_init, Sf_minus_init)
        init.append(Sf_init)
        init.append(t_init)

        self.plot_trajectory([X_init], ['init'])

        # self.opti.set_initial(sh, sh_init)
        self.opti.set_initial(X, X_init)
        self.opti.set_initial(Sf, Sf_init)
        self.opti.set_initial(S0, S0_init)
        self.opti.set_initial(U, U_init)
        self.opti.set_initial(T, t_init)

        self.opti.solver('ipopt', {'dump': True, 'dump_in': True, 'error_on_fail': False, 'ipopt': {
            "max_iter": 2000, 'hessian_approximation': 'exact', 'limited_memory_max_history': 5, 'print_level': 5}})
        # sol = self.opti.solve_limited()
        sol = self.opti.solve()

        print('optimal time is: ', sol.value(T), ' seconds')
        self.plot_trajectory([X_init, sol.value(X)], ['init', 'sol'])

        x0 = cs.vertcat(*init)
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
        pass

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
                # ind_count += self.n_sh
        ind_count += 1

        return cs.horzcat(*states_sol)

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

    def create_scaling_matrices(self, states=True, controls=True,
                                time=True, sep_hyp=False, slack0=False, slack_f=False):
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
            vec_sep_hyp = np.ones(self.n_sh)
        else:
            vec_sep_hyp = np.zeros(self.n_sh)

        if slack0:
            vec_slack0 = np.ones(self.n_s0)
        else:
            vec_slack0 = np.zeros(self.n_s0)

        if slack_f:
            vec_slack_f = np.ones(self.n_sf)
        else:
            vec_slack_f = np.zeros(self.n_sf)

        # Create the trust-region scaling matrix
        list_tr_scale_mat_vec = []
        list_tr_scale_mat_vec.append(vec_slack0)
        for k in range(self.N+1):
            list_tr_scale_mat_vec.append(vec_states)
            if k < self.N:
                list_tr_scale_mat_vec.append(vec_controls)
                # list_tr_scale_mat_vec.append(vec_sep_hyp)
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

    def convert_joints_to_end_effector(self, X):
        """
        This function gets a trajectory for the joints of the scara robot and
        transforms it into the cartesian trajectory of the end effector.
        """
        q1_sol = X[0, :]
        q3_sol = X[1, :]
        qd1_sol = X[2, :]
        qd3_sol = X[3, :]
        xee_sol = []
        yee_sol = []
        xee_dot_sol = []
        yee_dot_sol = []
        for i in range(self.N+1):
            q1 = q1_sol[i]
            q3 = q3_sol[i]
            qd1 = qd1_sol[i]
            qd3 = qd3_sol[i]
            # From joints q1 and q3 calculate the joints q2 and q4
            qv = self.qv_analytical_fun(cs.vertcat(q1, q3))
            q2 = qv[0]
            q4 = qv[1]
            q = cs.vertcat(q1, q2, q3, q4)

            # From torques qd1 and qd3 retrieve torques qd2 and qd4
            qdv = self.depinteg_fun(q, cs.vertcat(qd1, qd3))
            qd2 = qdv[0]
            qd4 = qdv[1]
            qd = cs.vertcat(qd1, qd2, qd3, qd4)

            # Calculates the position and velocity of the 'end effector'
            (pee_sol, _, vee_sol, _, _, _, _) = self.end_effector_fun(
                q, qd, cs.DM.zeros(q.shape[0]))

            # Retrieve position and speed of end effector
            xee = pee_sol[0]
            xee_sol.append(xee)
            yee = pee_sol[1]
            yee_sol.append(yee)

            xeed = vee_sol[0]
            xee_dot_sol.append(xeed)
            yeed = vee_sol[1]
            yee_dot_sol.append(yeed)

        return (xee_sol, yee_sol, xee_dot_sol, yee_dot_sol)

    def plot_trajectory(self, list_X, list_labels):
        """
        Plot the trajectory of some given joint trajectories. 
        The given trajectories are given as angles and torques on the joints 
        by list_X.
        """
        # Transform the angles and torques into the end effector positions
        list_xee_sol = []
        list_yee_sol = []
        list_xee_dot_sol = []
        list_yee_dot_sol = []
        for i in range(len(list_X)):
            (xee_sol,
             yee_sol,
             xee_dot_sol,
             yee_dot_sol) = self.convert_joints_to_end_effector(list_X[i])

            list_xee_sol.append(xee_sol)
            list_yee_sol.append(yee_sol)
            list_xee_dot_sol.append(xee_dot_sol)
            list_yee_dot_sol.append(yee_dot_sol)

        fig, ax = plt.subplots()
        X_wrksp, Y_wrksp = wspclim.workspace_boundary()
        ax.set_aspect('equal')
        rect = Rectangle(tuple(np.array(self.points[0, :]).squeeze(
        )), 0.02, 0.02, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.plot(X_wrksp, Y_wrksp)
        for i in range(len(list_X)):
            ax.plot(list_xee_sol[i], list_yee_sol[i], label=list_labels[i])
        ax.set_xlabel('$x_1$-axis')
        ax.set_ylabel('$x_2$-axis')
        ax.set_title('P2P-motion of scara robot')
        plt.legend()
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.show()

# # Process the solution
# q1_sol = sol.value(X)[0, :]
# q3_sol = sol.value(X)[1, :]
# qd1_sol = sol.value(X)[2, :]
# qd3_sol = sol.value(X)[3, :]

# # Solution of controls
# Qm1_sol = sol.value(U)[0, :]
# Qm2_sol = sol.value(U)[1, :]

# xee_sol = []
# yee_sol = []
# xeed_sol = []
# yeed_sol = []

# for i in range(N+1):
#     q1 = q1_sol[i]
#     q3 = q3_sol[i]
#     qd1 = qd1_sol[i]
#     qd3 = qd3_sol[i]
#     # From q1 and q3 calculate q2 and q4
#     qv = qv_analytical_fun(cs.vertcat(q1, q3))
#     q2 = qv[0]
#     q4 = qv[1]
#     q = cs.vertcat(q1, q2, q3, q4)

#     # From qd1 and qd3 retrieve
#     qdv = depinteg_fun(q, cs.vertcat(qd1, qd3))
#     qd2 = qdv[0]
#     qd4 = qdv[1]
#     qd = cs.vertcat(qd1, qd2, qd3, qd4)

#     # Calculates the position and velocity of the 'end effector'
#     (pee_sol, _, vee_sol, _, _, _, _) = end_effector_fun(
#         q, qd, cs.DM.zeros(q.shape[0]))

#     xee = pee_sol[0]
#     xee_sol.append(xee)
#     yee = pee_sol[1]
#     yee_sol.append(yee)

#     xeed = vee_sol[0]
#     xeed_sol.append(xeed)
#     yeed = vee_sol[1]
#     yeed_sol.append(yeed)

# # Plot the solution
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# rect = Rectangle(tuple(np.array(points[0, :]).squeeze(
# )), 0.02, 0.02, linewidth=1, edgecolor='r', facecolor='none')
# ax.add_patch(rect)
# ax.plot(xee_sol, yee_sol)
# ax.plot(X_wrksp, Y_wrksp)
# # ax.plot(p_load_init[:,0], p_load_init[:,1])
# # ax.set_xlim((-0.05, 0.7))
# # ax.set_ylim((-1, 0))
# plt.show()


# print('lol')
testproblem = scara_problem()
testproblem.create_problem()
