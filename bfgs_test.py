from classes.problem_class import optimisation_problem_class
from classes.method_class import optimisation_method_class
from classes.method_class import regular_newton
from classes.method_class import david_fletcher_powell
from classes.method_class import good_broyden
from classes.method_class import bad_broyden
from classes.method_class import symmetric_broyden
from classes.method_class import broyden_fletcher_goldfarb_shanno
from classes.hessian import hessian_approximation
from classes.inexact_line_search import inexact_line_search
from classes.exact_line_search import exact_line_search

def test_func(x):
    return x[0]**3 + x[0]*x[1] + (x[0]**2)*(x[1]**2) - 3*x[0]
def rosenbrock(x):
    return 100*((x[1] - x[0]*x[0])**2) + (1 - x[0])**2

optimization_problem = optimisation_problem_class(rosenbrock, h=1e-8)
line_search = exact_line_search(rosenbrock, optimization_problem.gradient_approx, tolerance = 1e-10)

BFGS = broyden_fletcher_goldfarb_shanno(optimization_problem, [-1.5, 1], 1e-10, line_search)
BFGS.find_min()
