import os
import pprint
import random
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import cm as colormap
import numpy as np

from math import log, sin, sqrt

from algorithms import *

figure = plt.figure()
ax = figure.gca(projection="3d")

epsilon = 0.1
A = Laplace(1/epsilon, 0)
def f(x, y):
    B = Laplace(A.spread, A.mean + x)
    C = Laplace(B.spread, B.mean + y)
    one = log(B.larger(A)/C.larger(A))
    two = log(A.larger(B)/A.larger(C))
    return max(abs(one), abs(two))
X = np.arange(-100,100,1)
Y = np.arange(-10,10,1)
X1, Y = np.meshgrid(X, Y)
Z = np.vectorize(f)(X1, Y)
ax.plot_surface(X1, Y, Z, cmap=colormap.viridis, linewidth=0, antialiased=False)

def g(x):
    return f(x, 1)
Y_ = np.ones(len(X))
Z_ = np.vectorize(g)(X)
ax.plot(X, Y_, zs=Z_, color="red")

ax.set_zlim(0,1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
