""" 
Do the Anderson Acceleration of the step
"""
# Import standard libraries
from __future__ import annotations  # <-still need this.
from typing import TYPE_CHECKING
import casadi as cs
import numpy as np
# Import self-written libraries
if TYPE_CHECKING:
    from fslp.nlp_problem import NLPProblem
    from fslp.options import Options

class AndersonAcceleration:

    def __init__(self, problem: NLPProblem, opts: Options):
        """ 
        Initialize the Anderson Acceleration
        """
        self.max_memory_size = opts.anderson_memory_size
        self.beta = opts.anderson_beta
        self.anderson_memory_step = cs.DM.zeros(problem.number_variables, self.max_memory_size)
        self.anderson_memory_iterate = cs.DM.zeros(problem.number_variables, self.max_memory_size)

        self.current_number_anderson_iterations = 0

    def init_memory(self, direction: cs.DM, iterate: cs.DM):
            """
            Initializes the memory of the Anderson acceleration.
            p       the step to be stored
            x       the iterate to be stored

            """
            # Probably not necessary
            # self.anderson_memory_step = cs.DM.zeros(self.nx, self.max_memory_size)
            # self.anderson_memory_iterate = cs.DM.zeros(self.nx, self.max_memory_size)

            self.current_number_anderson_iterations = 1

            # Should be the same for any m
            self.anderson_memory_step[:, 0] = direction
            self.anderson_memory_iterate[:, 0] = iterate

    def update_memory(self, direction: cs.DM, iterate: cs.DM):
        """
        Update the memory of the Anderson acceleration.
        p       the step to be stored
        x       the iterate to be stored

        """
        # Shift the memory one step further
        if self.max_memory_size != 1:
            self.anderson_memory_step[:,1:] = self.anderson_memory_step[:,0:-1]
            self.anderson_memory_iterate[:,1:] = self.anderson_memory_iterate[:,0:-1]
        
        # Increment the memory size
        self.current_number_anderson_iterations += 1

        # Is used for all updates
        self.anderson_memory_step[:, 0] = direction
        self.anderson_memory_iterate[:, 0] = iterate

    def step_update(self, current_direction: cs.DM, current_iterate: cs.DM):
        """
        This file does the Anderson step update

        Args:
            d_curr (cs.DM): Current direction (step)
            x_curr (cs.DM): Current iterate
        """
        if self.max_memory_size == 1:
            gamma = (current_direction.T @ (current_direction-self.anderson_memory_step[:,0]))/((current_direction-self.anderson_memory_step[:,0]).T @ (current_direction-self.anderson_memory_step[:,0]))
            x_plus = current_iterate + self.beta*current_direction - gamma*(current_iterate-self.anderson_memory_iterate[:,0] + self.beta*current_direction - self.beta*self.anderson_memory_step[:,0])
            print("gamma k: ", gamma)
        else:
            curr_stages = min(self.current_number_anderson_iterations, self.max_memory_size)

            p_stack = cs.horzcat(current_direction, self.anderson_memory_step[:, 0:curr_stages])
            x_stack = cs.horzcat(current_iterate, self.anderson_memory_iterate[:, 0:curr_stages])

            F_k = p_stack[:, 0:-1] - p_stack[:, 1:]
            # print('Dimension of F_k', F_k.shape)
            E_k = x_stack[:, 0:-1] - x_stack[:, 1:]

            pinv_Fk = np.linalg.pinv(F_k)
            gamma_k = pinv_Fk @ current_direction
            print("gamma k: ", gamma_k)
            x_plus = current_iterate + self.beta*current_direction -(E_k + self.beta*F_k) @ gamma_k
        
        # Always update the memory in the end
        self.update_memory(current_direction, current_iterate)

        return x_plus
    

    # def anderson_acc_init_memory(self, direction: cs.DM, iterate: cs.DM):
    #     """
    #     Initializes the memory of the Anderson acceleration.
    #     p       the step to be stored
    #     x       the iterate to be stored

    #     """
    #     self.anderson_memory_step = cs.DM.zeros(self.nx, self.max_memory_size)
    #     self.anderson_memory_iterate = cs.DM.zeros(self.nx, self.max_memory_size)

    #     # if self.sz_anderson_memory == 1:
    #     #     self.anderson_memory_step[:, 0] = p
    #     #     self.anderson_memory_iterate[:, 0] = x
    #     # else:
    #     #     raise NotImplementedError('Not implemented yet')

    #     # Should be the same for any m
    #     self.anderson_memory_step[:, 0] = direction
    #     self.anderson_memory_iterate[:, 0] = iterate

    # def anderson_acc_update_memory(self, p: cs.DM, x: cs.DM):
    #     """
    #     Update the memory of the Anderson acceleration.
    #     p       the step to be stored
    #     x       the iterate to be stored

    #     """
    #     # if self.sz_anderson_memory == 1:
    #     #     self.anderson_memory_step[:, 0] = p
    #     #     self.anderson_memory_iterate[:, 0] = x
    #     # else:
    #     if self.max_memory_size != 1:
    #         self.anderson_memory_step[:,1:] = self.anderson_memory_step[:,0:-1]
    #         self.anderson_memory_iterate[:,1:] = self.anderson_memory_iterate[:,0:-1]
    #         # raise NotImplementedError('Not implemented yet')

    #     # Is used for all updates
    #     self.anderson_memory_step[:, 0] = p
    #     self.anderson_memory_iterate[:, 0] = x

    # def anderson_acc_step_update(self, p_curr:cs.DM, x_curr:cs.DM, j:int):
    #     """
    #     This file does the Anderson step update

    #     Args:
    #         p (_type_): _description_
    #         x (_type_): _description_
    #         j: inner iterate index
    #     """
    #     beta = 1
    #     if self.sz_anderson_memory == 1:
    #         gamma = (p_curr.T @ (p_curr-self.anderson_memory_step[:,0]))/((p_curr-self.anderson_memory_step[:,0]).T @ (p_curr-self.anderson_memory_step[:,0]))
    #         x_plus = x_curr + beta*p_curr - gamma*(x_curr-self.anderson_memory_iterate[:,0] + beta*p_curr - beta*self.anderson_memory_step[:,0])
    #         # self.anderson_acc_update_memory(p_curr, x_curr)
    #     else:
    #         curr_stages = min(j, self.sz_anderson_memory)

    #         p_stack = cs.horzcat(p_curr, self.anderson_memory_step[:, 0:curr_stages])
    #         x_stack = cs.horzcat(x_curr, self.anderson_memory_iterate[:, 0:curr_stages])

    #         F_k = p_stack[:, 0:-1] - p_stack[:, 1:]
    #         # print('Dimension of F_k', F_k.shape)
    #         E_k = x_stack[:, 0:-1] - x_stack[:, 1:]

    #         pinv_Fk = np.linalg.pinv(F_k)
    #         gamma_k = pinv_Fk @ p_curr
    #         x_plus = x_curr + beta*p_curr -(E_k + beta*F_k) @ gamma_k
            
    #     self.anderson_acc_update_memory(p_curr, x_curr)

    #     return x_plus
