# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 16:15:51 2020

@author: Casper
"""
import numpy as np

class inexact_line_search:

    def __init__(self, func, grad, wolfe_powell = False):
        """
        Creates a class which performs an inexact line search.

        In:
            func: the objective function
            grad: its (possibly numerical) gradient

        Parameters and their default values:
            rho = 0.1
            sigma = 0.7
            tau = 0.1
            chi = 9.

        """
        self.wolfe_powell = wolfe_powell
        self.func = func
        self.grad = grad

        self.rho = 0.1 # Between 0, 0.5
        self.sigma = 0.7 # rho < sigma =< 1
        self.tau = 0.1
        self.chi = 9.


    def __call__(self, x_0, s):
        """
        Given an initial guess, returns a factor for use in line search which
        is acceptable according to either Goldstein or Wolfe-Powell
        conditions.

        In:
            x_0: the starting point
            s: the direction along which to line search
            guess: initial guess for line search factor alpha_0
            wolfe_powell: optional boolean, controls whether to use Goldstein
            or Wolfe-Powell conditions (recommended for non-quadratic
            objective function)
        Out:
            a0: scalar, line search factor alpha_0
            f_a0: scalar, the objective function evaluated for x + a0*s

        """
        self.s = s
        self.func_alpha = lambda alpha: self.func(x_0 + alpha*s)
        self.grad_alpha = lambda alpha: self.grad(x_0 + alpha*s)


        self.a0 = 1
        self.aL = 0
        self.aU = 1e+40

        self.update_function_values()

        while not (self.left_cond(self.wolfe_powell) and self.right_cond()):
            if not self.left_cond(self.wolfe_powell):
                self.extrapolation()
                self.update_function_values(new_aL = True)
            else:
                self.interpolation()
                self.update_function_values(new_aL = False)

        return self.a0, self.f_a0





    def left_cond(self, wolfe_powell = False):
        """
        Checks if current a0 is an acceptable point in the lower bound, using
        either Goldstein (default) or Wolfe-Powell conditions (recommended if
        objective function is non-quadratic)

        In:
            self: calls current values of a0, aL, f_aL and such.
            wolfe_powell: boolean, controls which conditions to use.
        Out:
            boolean, True if a0 is an acceptable point

        """
        if wolfe_powell == False:
            RHS = self.f_aL + (1 - self.rho)*(self.a0 - self.aL)*self.g_aL
            cond = self.f_a0 >= RHS
        else:
            cond = self.g_a0 >= self.sigma*self.g_aL
        return cond

    def right_cond(self):
        """
        Checks if current a0 is an acceptable point in the upper bound. Note
        that compared to the lower bound, this formula is identical for
        both Goldstein and Wolfe-Powell conditions.

        In:
            self, calls current values of a0, aL, f_aL and such.
        Out:
            boolean, True if a0 is an acceptable point

        """
        RHS = self.f_aL + self.rho*(self.a0 - self.aL)*self.g_aL
        cond = self.f_a0 <= RHS
        return cond

    def extrapolation(self):
        """
        Calculates and updates values for a0 and aL using extrapolation.

        In:
            self, calls current values of a0, aL, f_aL and such.
        Out:
            returns nothing, updates a0 and aL

        """
        zero_factor = self.a0 - self.aL
        max_factor = self.tau*zero_factor
        min_factor = self.chi*zero_factor
        # Extrapolation step
        delta_a0 = zero_factor*self.g_a0/(self.g_aL - self.g_a0)
        # Compare to current values
        if delta_a0 < max_factor:
            delta_a0 = max_factor

        if delta_a0 > min_factor:
            delta_a0 = min_factor
        # Update variable
        self.aL = self.a0
        self.a0 = self.a0 + delta_a0

    def interpolation(self):
        """
        Calculates and updates the value for a0 using interpolation.

        In:
            self, calls current values of a0, aL, f_aL and such.
        Out:
            returns nothing
            updates a0 and aU

        """
        # Update aU if a0 is smaller
        if self.a0 < self.aU:
            self.aU = self.a0

        upper_factor = self.aU - self.aL
        zero_factor = self.a0 - self.aL
        # Interpolation step
        denominator = 2*(self.f_aL - self.f_a0 + zero_factor*self.g_aL)
        new_a0 = zero_factor**2*self.g_aL/denominator

        max_factor = self.aL + self.tau*upper_factor
        min_factor = self.aU - self.tau*upper_factor
        # Compare to current values
        if new_a0 < max_factor:
            new_a0 = max_factor

        if new_a0 > min_factor:
            new_a0 = min_factor
        # Update a0
        self.a0 = new_a0

    def update_function_values(self, new_aL = True):
        """
        Evaluates f_alpha and g_alpha for new parameters a0 and aL.

        In:
            self: calls current values of a0, aL and such.
            new_aL: boolean, can optionally be set to False if new evaluation
            for aL is unwanted (such as when aL has not been updated).
        Out:
            returns nothing
            updates f_a0 and g_a0
            if new_aL is set to True, also updates f_aL and g_aL


        """
        self.f_a0 = self.func_alpha(self.a0)
        self.g_a0 = np.dot(self.grad_alpha(self.a0), self.s)
        if new_aL == True:
            self.f_aL = self.func_alpha(self.aL)
            self.g_aL = np.dot(self.grad_alpha(self.aL), self.s)
