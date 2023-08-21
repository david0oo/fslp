""" 
Do the Anderson Acceleration of the step
"""

import casadi as cs
import numpy as np


class AndersonAcceleration:

    def __init__(self, nx: int, memory_size: int = 1, beta: float = 1):
        """ 
        Initialize the Anderson Acceleration
        """
        self.max_memory_size = memory_size
        self.beta = beta
        self.nx = nx
        self.anderson_memory_step = cs.DM.zeros(self.nx, self.max_memory_size)
        self.anderson_memory_iterate = cs.DM.zeros(self.nx, self.max_memory_size)

        self.curr_memory_size = 0

    def init_memory(self, p: cs.DM, x: cs.DM):
            """
            Initializes the memory of the Anderson acceleration.
            p       the step to be stored
            x       the iterate to be stored

            """
            # Probably not necessary
            # self.anderson_memory_step = cs.DM.zeros(self.nx, self.max_memory_size)
            # self.anderson_memory_iterate = cs.DM.zeros(self.nx, self.max_memory_size)

            self.curr_memory_size = 0

            # Should be the same for any m
            self.anderson_memory_step[:, 0] = p
            self.anderson_memory_iterate[:, 0] = x

    def update_memory(self, p:cs.DM, x:cs.DM):
        """
        Update the memory of the Anderson acceleration.
        p       the step to be stored
        x       the iterate to be stored

        """
        if self.max_memory_size != 1:
            self.anderson_memory_step[:,1:] = self.anderson_memory_step[:,0:-1]
            self.anderson_memory_iterate[:,1:] = self.anderson_memory_iterate[:,0:-1]
            # Increment the memory size
            self.curr_memory_size += 1

        # Is used for all updates
        self.anderson_memory_step[:, 0] = p
        self.anderson_memory_iterate[:, 0] = x

    def __call__(self, d_curr:cs.DM, x_curr:cs.DM):
        """
        This file does the Anderson step update

        Args:
            d_curr (cs.DM): Current direction (step)
            x_curr (cs.DM): Current iterate
        """
        if self.curr_memory_size == 0:
            self.init_memory(d_curr, x_curr)
            # Put initial iterate into memory
            x_plus = x_curr + d_curr
            self.curr_memory_size += 1
        else:
            if self.max_memory_size == 1:
                gamma = (d_curr.T @ (d_curr-self.anderson_memory_step[:,0]))/((d_curr-self.anderson_memory_step[:,0]).T @ (d_curr-self.anderson_memory_step[:,0]))
                x_plus = x_curr + self.beta*d_curr - gamma*(x_curr-self.anderson_memory_iterate[:,0] + self.beta*d_curr - self.beta*self.anderson_memory_step[:,0])
                print("gamma k: ", gamma)
            else:
                curr_stages = min(self.curr_memory_size, self.max_memory_size)

                p_stack = cs.horzcat(d_curr, self.anderson_memory_step[:, 0:curr_stages])
                x_stack = cs.horzcat(x_curr, self.anderson_memory_iterate[:, 0:curr_stages])

                F_k = p_stack[:, 0:-1] - p_stack[:, 1:]
                # print('Dimension of F_k', F_k.shape)
                E_k = x_stack[:, 0:-1] - x_stack[:, 1:]

                pinv_Fk = np.linalg.pinv(F_k)
                gamma_k = pinv_Fk @ d_curr
                print("gamma k: ", gamma_k)
                x_plus = x_curr + self.beta*d_curr -(E_k + self.beta*F_k) @ gamma_k
            
            self.update_memory(d_curr, x_curr)

        return x_plus
    

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
