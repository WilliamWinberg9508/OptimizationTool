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


#Some functions to test the newton method on
def g(x):
    return x[0]*x[0] + x[1]*x[1]
def f(x):
    return
def rosenbrock(x):
    return 100*((x[1] - x[0]*x[0])**2) + (1 - x[0])**2

def rgrad(x):
    return [400*x[0]- 400*x[0]*x[1] + 2*x[0] - 2, 200*(x[1]-x[0]*x[0])]

def test_func(x):
    return x[0]**3 + x[0]*x[1] + (x[0]**2)*(x[1]**2) - 3*x[0]


print('Here we use the regular newton with exact line search on the rosenbrock function, x_0 = [2, 2]')
input('Press enter to continue')

optimization_problem = optimisation_problem_class(rosenbrock, h=1e-8)
line_search = exact_line_search(rosenbrock, optimization_problem.gradient_approx, tolerance = 1e-10)
regular_newton = regular_newton(optimization_problem, [2,2], 1e-10, line_search)

regular_newton.find_min()


print('Here we use the david_fletcher_powell with exact line search on the rosenbrock function x_0 = [1.5, 1.5]')
input('Press enter to continue')

optimization_problem = optimisation_problem_class(rosenbrock, h=1e-5)
line_search = exact_line_search(rosenbrock, optimization_problem.gradient_approx, tolerance = 1e-7)
DFP = david_fletcher_powell(optimization_problem, [1.5,1.5], 1e-5, line_search)
DFP.find_min(iterations=3)


print('Here we use the good broyden with exact line search on f = x^3 + x*y +(x^2)*(y^2) - 3y, (x,y) = [1.1, -0.6]')
input('Press enter to continue')

optimization_problem = optimisation_problem_class(test_func, h=1e-13)
line_search = exact_line_search(test_func, optimization_problem.gradient_approx, tolerance = 1e-10)

GB = good_broyden(optimization_problem, [1.1, -.6], 1e-10, line_search)
print(GB.find_min()[-1])

print('Here we use the bad broyden with exact line search on f = x^3 + x*y +(x^2)*(y^2) - 3y, (x,y) = [1.1, -0.6]')
input('Press enter to continue')

optimization_problem = optimisation_problem_class(test_func, h=1e-3)
line_search = exact_line_search(test_func, optimization_problem.gradient_approx, tolerance = 1e-10)

BB = bad_broyden(optimization_problem, [1.1, -.6], 1e-5, line_search)
print(BB.find_min()[-1])

print('Here we use the symmetric broyden with exact line search on f = x^3 + x*y +(x^2)*(y^2) - 3y, (x,y) = [1.1, -0.6]')
input('Press enter to continue')

optimization_problem = optimisation_problem_class(test_func, h=1e-3)
line_search = exact_line_search(test_func, optimization_problem.gradient_approx, tolerance = 1e-10)

SB = symmetric_broyden(optimization_problem, [1.1, -.6], 1e-5, line_search)
print(SB.find_min()[-1])

print('Here we use the BFGS with exact line search on f = x^3 + x*y +(x^2)*(y^2) - 3y, (x,y) = [1.1, -0.6]')
input('Press enter to continue')

optimization_problem = optimisation_problem_class(test_func, h=1e-3)
line_search = exact_line_search(test_func, optimization_problem.gradient_approx, tolerance = 1e-10)

BFGS = broyden_fletcher_goldfarb_shanno(optimization_problem, [1.1, -.6], 1e-5, line_search)
print(BFGS.find_min()[-1])
