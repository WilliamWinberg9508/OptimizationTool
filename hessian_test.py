#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 15:17:03 2020

@author: halli
"""

from scipy import*
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import classes

"""
Test stuff
"""

def f(x):
    return x[0]**3 + x[1]**2 - 4*x[2]**4 + x[3]*x[1]


x_bar = [0.5, 0.7, 0.9, 1.0]

#grad = gradient(f,x_bar,0.1)

def g(x):
    return x[0]**3 + x[1]**4

x_bar_2 = [2,3]

H = classes.hessian_approximation(f)
print(H(x_bar,0.01))

H2 = classes.hessian_approximation(g)
print(H2(x_bar_2,0.01))
