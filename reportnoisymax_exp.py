import os
import pprint
import random
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import cm as colormap
import matplotlib.ticker as ticker
import numpy as np

from math import log, sin, sqrt

from algorithms import *

figure = plt.figure()
ax = figure.gca(projection="3d")

epsilon = 0.1


def f(x, y):
    A = 1  # == exp(epsilon * 0 / 2), since we fix one query at 0
    norm_x = 1 / (1 + exp(epsilon * x))
    B = norm_x * exp(epsilon * x)
    A_b = norm_x * A

    norm_xy = 1 / (1 + exp(epsilon * (x+y)))
    C = norm_xy * exp(epsilon * (x + y))
    A_c = norm_xy * A

    # differential probability of largest query being reported as maximal
    one = log(B/C)
    # also take into account what happens with the second-largest one in the differential case!
    two = log(A_b/A_c)
    return max(abs(one), abs(two))

X = np.arange(-100, 100, 1)
Y = np.arange(-10, 11, 1)
X1, Y = np.meshgrid(X, Y)
Z = np.vectorize(f)(X1, Y)
ax.plot_surface(X1, Y, Z, cmap=colormap.viridis, linewidth=0, antialiased=False)


def g(x):
    return f(x, 1)

Y_ = np.ones(len(X))
Z_ = np.vectorize(g)(X)
ax.plot(X, Y_, zs=Z_, color="red")
print(max(Z_))

ax.set_zlim(0, 1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.azim = 45
ax.elev = 10
plt.show()
