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
from chebyquad_problem import *

x=linspace(0,1,11)
optimization_problem = optimisation_problem_class(chebyquad, gradient=gradchebyquad, h=1e-10)
line_search = inexact_line_search(chebyquad, gradchebyquad, True)
regular_newton = regular_newton(optimization_problem, x, 1e-6, line_search)

x_opt = regular_newton.find_min()[-1]

x=linspace(0,1,11)
xmin= so.fmin_bfgs(chebyquad,x,gradchebyquad)  # should converge after 18 iterations
#x_opt.sort()
#xmin.sort()
print("This is our methods result")
print(x_opt)
print("This is scipy's results")
print(xmin)
print("This is the evaluation at our x")
print(chebyquad(x_opt))
print("This is the evaluation at their x")
print(chebyquad(xmin))
