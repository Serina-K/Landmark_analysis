# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 15:19:59 2018

@author: serina
"""

import numpy as np




obs = np.loadtxt('all.txt')
obs = obs.reshape((-1, 1))

#----------#

import matplotlib.pyplot as plt

num_bins = 50
n, bins, patches = plt.hist(obs, num_bins, normed=1)
plt.savefig("Data_all.png")

#----------#

from sklearn import mixture

g = mixture.GaussianMixture(n_components=2, covariance_type='spherical')

g.fit(obs)

weights = g.weights_
means = g.means_
covars = g.covariances_

# print(round(weights[0],2), round(weights[1],2))
# print(round(means[0][0],2), round(means[1][0],2))
# print(round(covars[0],2), round(covars[1],2))

#----------#

import math
import scipy.stats as stats

D = obs.ravel()
xmin = D.min()
xmax = D.max()
x = np.linspace(xmin,xmax,1000)

mean1 = means[0]
sigma1 = math.sqrt(covars[0])
gauss1 = weights[0]*stats.norm.pdf(x, mean1, sigma1)
plt.plot(x, gauss1, c='red')

mean2 = means[1]
sigma2 = math.sqrt(covars[1])
gauss2 = weights[1]*stats.norm.pdf(x, mean2, sigma2)
plt.plot(x, gauss2, c='blue')

ind0 = np.argmin(abs(gauss1[100:700] - gauss2[100:700])) + 100

plt.plot(x[ind0], gauss1[ind0], 'o')
print(x[ind0], gauss1[ind0])


plt.savefig("DataGMM_all.png")
plt.show()