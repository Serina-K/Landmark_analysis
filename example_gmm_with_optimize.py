#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 14:12:17 2018

@author: zeynep
"""

import numpy as np
import matplotlib.pyplot as plt


from scipy import optimize
from scipy.stats import norm

def eval_norm(x, m, s):
    """
    Evaluate normal distribution for given x with given mean m and std s
    """
    return 1/(2*np.pi*s**2)*np.exp(-0.5*(x-m)**2/s**2)

def true_fun(x, s1, s2, w1, w2):
    """
    True function is a mixture of two gaussians:
        w1*N(m1,s2) + w2*N(m2, s2)
    """
    m1 = -0.5
    m2 = 1
    return w1*eval_norm(x, m1, s1) + w2*eval_norm(x, m2, s2)

def gen_dist():
    """
    This funtion generates a sample from the ditrubtion coming from the
    true function. but in doing that, it does not use directly the above true_fun.
    """
    mu1, sigma1 = -0.5, 0.2 # mean and standard deviation
    mu2, sigma2 = 1, 0.5 # mean and standard deviation
    w1, w2 = 0.5, 2 # these ae amplitudes (sample sizes below)
    g1 = np.random.normal(mu1, sigma1, 500)
    g2 = np.random.normal(mu2, sigma2, 2000)
    gd = np.concatenate((g1, g2), axis=0)
    count, bins, ignored = plt.hist(gd, 100, density=True)
    return count, bins


count, bins = gen_dist()
x_data = bins[0:-1]
y_data = count

params, params_covariance = optimize.curve_fit(true_fun, x_data, y_data,
                                               p0=[0.2,0.2, 0.5, 0.5])


plt.plot(x_data, true_fun(x_data, params[0], params[1], params[2], params[3]),
         label='Fitted function')


plt.grid(linestyle='dotted')
plt.show()