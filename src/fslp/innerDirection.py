import casadi as cs

class InnerDirection:

    def __init__(self) -> None:
        pass

    def anderson_acc_init_memory(self, p:cs.DM, x:cs.DM):
        """
        Initializes the memory of the Anderson acceleration.
        p       the step to be stored
        x       the iterate to be stored

        """
        self.anderson_memory_step = cs.DM.zeros(self.nx, self.sz_anderson_memory)
        self.anderson_memory_iterate = cs.DM.zeros(self.nx, self.sz_anderson_memory)

        # if self.sz_anderson_memory == 1:
        #     self.anderson_memory_step[:, 0] = p
        #     self.anderson_memory_iterate[:, 0] = x
        # else:
        #     raise NotImplementedError('Not implemented yet')

        # Should be the same for any m
        self.anderson_memory_step[:, 0] = p
        self.anderson_memory_iterate[:, 0] = x

    def anderson_acc_update_memory(self, p:cs.DM, x:cs.DM):
        """
        Update the memory of the Anderson acceleration.
        p       the step to be stored
        x       the iterate to be stored

        """
        # if self.sz_anderson_memory == 1:
        #     self.anderson_memory_step[:, 0] = p
        #     self.anderson_memory_iterate[:, 0] = x
        # else:
        if self.sz_anderson_memory != 1:
            self.anderson_memory_step[:,1:] = self.anderson_memory_step[:,0:-1]
            self.anderson_memory_iterate[:,1:] = self.anderson_memory_iterate[:,0:-1]
            # raise NotImplementedError('Not implemented yet')

        # Is used for all updates
        self.anderson_memory_step[:, 0] = p
        self.anderson_memory_iterate[:, 0] = x

    def anderson_acc_step_update(self, p_curr:cs.DM, x_curr:cs.DM, j:int):
        """
        This file does the Anderson step update

        Args:
            p (_type_): _description_
            x (_type_): _description_
            j: inner iterate index
        """
        beta = 1
        if self.sz_anderson_memory == 1:
            gamma = (p_curr.T @ (p_curr-self.anderson_memory_step[:,0]))/((p_curr-self.anderson_memory_step[:,0]).T @ (p_curr-self.anderson_memory_step[:,0]))
            x_plus = x_curr + beta*p_curr - gamma*(x_curr-self.anderson_memory_iterate[:,0] + beta*p_curr - beta*self.anderson_memory_step[:,0])
            # self.anderson_acc_update_memory(p_curr, x_curr)
        else:
            curr_stages = min(j, self.sz_anderson_memory)

            p_stack = cs.horzcat(p_curr, self.anderson_memory_step[:, 0:curr_stages])
            x_stack = cs.horzcat(x_curr, self.anderson_memory_iterate[:, 0:curr_stages])

            F_k = p_stack[:, 0:-1] - p_stack[:, 1:]
            # print('Dimension of F_k', F_k.shape)
            E_k = x_stack[:, 0:-1] - x_stack[:, 1:]

            pinv_Fk = np.linalg.pinv(F_k)
            gamma_k = pinv_Fk @ p_curr
            x_plus = x_curr + beta*p_curr -(E_k + beta*F_k) @ gamma_k
            
        self.anderson_acc_update_memory(p_curr, x_curr)

        return x_plus


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
