

class exact_line_search:

    def __init__(self, func, grad, tolerance):
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
        self.func = func
        self.grad = grad
        self.tolerance = tolerance


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
        """
        Exact line search using the bisection method
        In: x is the current point, dir is the search direction
        Out: returnes the stepsize the minimized the search function
        """
        #This is an implementation of the bisection method for exact line search

        #This is the derivative of our F(lambda) = f(x +lambda*dir). The search function is minimized when this is equal to 0.
        search_func = lambda step: (self.grad(x_0 + step*s) @ s)

        #We create an interval starting at the derivative of the current point
        a = 0
        #a = search_func(0);
        step_size = 0.01
        b = a
        #Search in the search direction until the sign changes, then the minimum is within
        #this interval, SEEMS LIKE WE SOMETIME SEARCH IN A NON DESCENT DIRECTION WHICH MAKES THIS LOOP RUN ENDLESSLY
        while search_func(a)*search_func(b) > 0:
            step_size = step_size*2
            b = step_size
            #b = search_func(step_size)

        #Halve the interval until it has reached a certain length

        while abs(b-a) > self.tolerance:
            m = (a + b)/2
            f_m = search_func(m)

            if f_m <= 0:
                a = m
            elif f_m > 0:
                b = m
            else:
                print("Bisection method fails")
                return
        #Return the midpoint of the interval
        m = (a+b)/2
        return m, search_func(m)
