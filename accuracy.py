import os
import pprint
import random

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import cm as colormap
import numpy as np

from math import log, sin, sqrt

from algorithms import *

figure = plt.figure()
ax = figure.gca()

def accuracy(k, b, e):
	return 8*(log(k) + log(2/b))/e


b = 0.1
e_T = 0.1
e_Q = 0.01
a = accuracy(k=1, b=b, e=e_T)
T = Laplace(1/e_T, 0)
def f(x):
    Q = Laplace(1/e_Q, x)
    return Q.larger(T)

X = np.arange(-2*a,2*a,1)
Y = np.vectorize(f)(X)
ax.plot(X, Y)
ax.axvline(x=a, color="red")
ax.axvline(x=-a, color="red")
ax.axhline(y=1-b, color="green")
ax.axhline(y=b, color="green")


plt.show()
