from scipy import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from .hessian import hessian_approximation
from .problem_class import optimisation_problem_class

class optimisation_method_class:
    """
    This is the base optimization class to be extended by other classes
    takes an optimisation_problem_class which describes the objective function: optimisation_problem_class,
     an initial point: x_0,
     a line search object,
     and a convergence tolerance: tolerance
    """
    def __init__(self, optimisation_problem_class, x_0, tolerance, line_search):
        self.optimisation_problem_class = optimisation_problem_class
        self.x_0 = x_0
        self.tolerance = tolerance
        self.line_search = line_search

    """
    This is the method you shold call on the children of this class to find the minimum using
    the specific methods defined in the child class.
    """
    def find_min(self):
        cond = True
        x = self.x_0
        x_all = []

        while cond:
            x_all.append(x)
            print(x)

            x = self.update_x(x)

            if la.norm(x - x_all[-1]) < self.tolerance:
                x_all.append(x)
                print(x)
                cond = False
        return x_all
    """This is the method that updates x using the stepsize and search direction
    obtained from the specific methods"""
    def update_x(self, x):

        dir = self.search_dir(x)
        a = self.line_search_factor(x, dir)

        return x + a[0]*dir

    """This returnes the stepsize determined either by inexact line search or exact line search"""
    def line_search_factor(self):
        return 1
    """This returns the search direction s given by the specified method"""
    def search_dir(self, x):
        return 1
    """This updates the hessian approximation for the given quasi-newton method"""
    def update_hessian(self, x, x_old):
        return 1

class regular_newton(optimisation_method_class):

    def __init__(self, optimisation_problem_class, x_0, tol, line_search):
        optimisation_method_class.__init__(self, optimisation_problem_class, x_0, tol, line_search)

    #Calculates the line search factor, maybe take more parameters to decide if we do inexact or exact, or no line search
    def line_search_factor(self, x, dir):
        return self.line_search(x, dir)
    #Calculates the newton direction using approximation methods contained in hessian.py
    def search_dir(self, x):
        gradient = np.array(self.optimisation_problem_class.gradient_approx(x))

        hessian = np.array(self.optimisation_problem_class.hessian_approx(x))

        return -np.linalg.inv(hessian) @ gradient

class good_broyden(optimisation_method_class):

    def __init__(self, optimisation_problem_class, x_0, tol, line_search):
        optimisation_method_class.__init__(self, optimisation_problem_class, x_0, tol, line_search)
        self.H_estimate = np.eye(len(x_0))

    #Calculates the line search factor, maybe take more parameters to decide if we do inexact or exact, or no line search
    def line_search_factor(self, x, dir):
        return self.line_search(x, dir)
    #Calculates the newton direction using approximation methods contained in hessian.py
    def search_dir(self, x):
        gradient = np.array(self.optimisation_problem_class.gradient_approx(x))

        return -self.H_estimate @ gradient

    def update_x(self, x):

        dir = self.search_dir(x)
        a = self.line_search_factor(x, dir)
        x_new = x + a[0]*dir

        self.update_hessian(x_new, x)

        return x_new

    def update_hessian(self, x, x_old):

        g_old = np.array(self.optimisation_problem_class.gradient_approx(x_old))
        g_new = np.array(self.optimisation_problem_class.gradient_approx(x))
        delta = x - x_old
        gamma = g_new - g_old
        H = self.H_estimate
        self.H_estimate = H + ((delta - H@gamma)@(H@delta).T)/(H@delta.T @ gamma)

class bad_broyden(optimisation_method_class):

    def __init__(self, optimisation_problem_class, x_0, tol, line_search):
        optimisation_method_class.__init__(self, optimisation_problem_class, x_0, tol, line_search)
        self.H_estimate = np.eye(len(x_0))

    #Calculates the line search factor, maybe take more parameters to decide if we do inexact or exact, or no line search
    def line_search_factor(self, x, dir):
        return self.line_search(x, dir)
    #Calculates the newton direction using approximation methods contained in hessian.py
    def search_dir(self, x):
        gradient = np.array(self.optimisation_problem_class.gradient_approx(x))

        return -self.H_estimate @ gradient

    def update_x(self, x):

        dir = self.search_dir(x)
        a = self.line_search_factor(x, dir)
        x_new = x + a[0]*dir

        self.update_hessian(x_new, x)

        return x_new

    def update_hessian(self, x, x_old):

        g_old = np.array(self.optimisation_problem_class.gradient_approx(x_old))
        g_new = np.array(self.optimisation_problem_class.gradient_approx(x))
        delta = x - x_old
        gamma = g_new - g_old
        H = self.H_estimate
        self.H_estimate = H + np.outer((delta - H@gamma)/(np.inner(gamma,gamma)),gamma)

class symmetric_broyden(optimisation_method_class):

    def __init__(self, optimisation_problem_class, x_0, tol, line_search):
        optimisation_method_class.__init__(self, optimisation_problem_class, x_0, tol, line_search)
        self.H_estimate = np.eye(len(x_0))

    #Calculates the line search factor, maybe take more parameters to decide if we do inexact or exact, or no line search
    def line_search_factor(self, x, dir):
        return self.line_search(x, dir)
    #Calculates the newton direction using approximation methods contained in hessian.py
    def search_dir(self, x):
        gradient = np.array(self.optimisation_problem_class.gradient_approx(x))

        return -self.H_estimate @ gradient

    def update_x(self, x):

        dir = self.search_dir(x)
        a = self.line_search_factor(x, dir)
        x_new = x + a[0]*dir

        self.update_hessian(x_new, x)

        return x + a[0]*dir

    def update_hessian(self, x, x_old):

        g_old = np.array(self.optimisation_problem_class.gradient_approx(x_old))
        g_new = np.array(self.optimisation_problem_class.gradient_approx(x))
        delta = x - x_old
        gamma = g_new - g_old
        H = self.H_estimate
        u = delta - H@gamma
        a = 1/np.inner(u,gamma)
        self.H_estimate = H + a*np.outer(u,u)

class broyden_fletcher_goldfarb_shanno(optimisation_method_class):

    def __init__(self, optimisation_problem_class, x_0, tol, line_search):
        optimisation_method_class.__init__(self, optimisation_problem_class, x_0, tol, line_search)
        self.H_estimate = np.eye(len(x_0))

    #Calculates the line search factor, maybe take more parameters to decide if we do inexact or exact, or no line search
    def line_search_factor(self, x, dir):
        return self.line_search(x, dir)
    #Calculates the newton direction using approximation methods contained in hessian.py
    def search_dir(self, x):
        gradient = np.array(self.optimisation_problem_class.gradient_approx(x))

        return -self.H_estimate @ gradient

    def update_x(self, x):

        dir = self.search_dir(x)
        a = self.line_search_factor(x, dir)
        x_new = x + a[0]*dir

        self.update_hessian(x_new, x)

        return x_new

    def update_hessian(self, x, x_old):
        g_old = np.array(self.optimisation_problem_class.gradient_approx(x_old))
        g_new = np.array(self.optimisation_problem_class.gradient_approx(x))
        delta = x - x_old
        gamma = g_new - g_old
        H = self.H_estimate
        self.H_estimate = H + (1 + (gamma.T@H@gamma)/(np.inner(delta,gamma)))*((np.outer(delta,delta)/np.inner(delta,gamma))) - (np.outer(delta,gamma)@H + H@np.outer(gamma,delta))/(np.inner(delta,gamma))

        #print(la.norm(self.H_estimate - np.linalg.inv(self.exact_hessian(x))))

    #def exact_hessian(self, x):
    #    return np.array([[-400*(x[1] - x[0]**2) + 800*x[0]**2 + 2, -400*x[0]],[-400*x[0], 200]])
"""
This is the DFP optimization algorith that extends the general optimisation_method_class
Unfortunatly I had to update the find_min method since the iteration step is not the same
"""
class david_fletcher_powell(optimisation_method_class):

    def __init__(self, optimisation_problem_class, x_0, tol, line_search):
        optimisation_method_class.__init__(self, optimisation_problem_class, x_0, tol, line_search)
        self.D = np.identity(len(x_0))

    #We have to overrite the find_min function to contain the second loop
    #h is for estimating derivatives (close to 0 relative to the scale of the function)
    #iterations is the number of times you want to run the second iteration loop.
    def find_min(self, iterations=2):
        cond = True
        x = self.x_0
        x_all = []

        while cond:
            x_all.append(x)
            print(x)
            x = self.update_x(x, iterations)

            if la.norm(x - x_all[-1]) < self.tolerance:
                cond = False
                x_all.append(x)
                print(x)

        return x_all

    def update_x(self, x, iterations):
        self.D = np.identity(len(x))
        y = x
        for i in range(iterations):
            d_i = self.search_dir(y)
            a = self.line_search_factor(y, d_i)


            y_old = y
            y = y + a[0]*d_i
            self.update_D(y, y_old)
        return y


    #Calculates the line search factor, maybe take more parameters to decide if we do inexact or exact, or no line search
    def line_search_factor(self, x, dir):
        return self.line_search(x, dir)
    #Calculates the newton direction using approximation methods contained in hessian.py
    def search_dir(self, x):
        gradient = np.array(self.optimisation_problem_class.gradient_approx(x))
        return -self.D @ gradient

    #This method updates the matrix D in each step of the inner loop of the DFP algorithm
    def update_D(self, y, y_old):

        #Both p and q close to zero sometimes, this is not that good
        p = np.array(np.subtract(y, y_old))

        q = np.array(self.optimisation_problem_class.gradient_approx(y)) - \
            np.array(self.optimisation_problem_class.gradient_approx(y_old))

        first_term = 1/(p @ q)*(p @ p)

        second_term_first = 1 / (q @ self.D @ q)

        second_term_second = self.D @ q @ q

        second_term_final = second_term_first*second_term_second*self.D

        self.D = self.D + first_term - second_term_final


        return
