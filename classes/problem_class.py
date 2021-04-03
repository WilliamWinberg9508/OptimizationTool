from .hessian import hessian_approximation
class optimisation_problem_class:
    """Function is the objective function to be evaluated,
    h is used to approximate the gradient and the hessian, should be small (close to zero) relative to the function scale,
    gradient is an optional term if the gradient is actually given"""
    def __init__(self, function, h=1e-5, gradient=None):
        self.function = function
        self.hessian_approximation = hessian_approximation(self.function)
        self.gradient = gradient
        self.h = h
    #Evaluates the objective function at given point x
    def __call__(self, x):
        return self.function(x)

    #Approximates the gradient at x, where h is small number used to approximate derivative
    #If gradient was given, instead evaluates directly
    def gradient_approx(self, x):

        if self.gradient == None:
            return self.hessian_approximation.gradient(self.function, x, self.h)
        else:
            return self.gradient(x)
    #Approximates the hessian at x, where h is small number used to approximate derivative
    def hessian_approx(self, x):
        return self.hessian_approximation(x, self.h)
