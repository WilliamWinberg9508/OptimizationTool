#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:51:57 2020

@author: halli
"""

import numpy as np

class hessian_approximation:

    def __init__(self,f,gradient=False):
        self.f = f

    def __call__(self,x,h):
        return self.hessian(x,h)


    def gradient(self,f,x,h):
        """

        Parameters
        ----------
        f : TYPE
            DESCRIPTION.
        x_bar : TYPE
            DESCRIPTION.
        h : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        g = []
        for i in range(len(x)):
            e_basis = np.zeros(len(x))
            e_basis[i] = h
            a = (f(x + e_basis) - f(x))/h
            g.append(a)
        return np.asarray(g)


    def derivative_approx(self,x, h, i):
        """


        Parameters
        ----------
        f : TYPE
            DESCRIPTION.
        x : TYPE
            DESCRIPTION.
        h : TYPE
            DESCRIPTION.
        i : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        e_basis = np.zeros(len(x))
        e_basis[i] = h
        return lambda x: (self.f(x+e_basis) - self.f(x))/h

    def hessian(self,x,h):
        """


        Parameters
        ----------
        f : TYPE
            DESCRIPTION.
        x : TYPE
            DESCRIPTION.
        h : TYPE
            DESCRIPTION.

        Returns
        -------
        H : TYPE
            DESCRIPTION.

        """
        H = self.gradient(self.derivative_approx(x, h, 0), x, h)
        for i in range(1,len(x)):
            H = np.vstack((H, self.gradient(self.derivative_approx(x,h,i), x,h)))
        return H
