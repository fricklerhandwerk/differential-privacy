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
    # return max(abs(one), abs(two))
    return one
X = np.arange(-150,150,1)
Y = np.arange(-5,5,0.25)
X, Y = np.meshgrid(X, Y)

Z = np.vectorize(f)(X, Y)
surface = ax.plot_surface(X, Y, Z, cmap=colormap.viridis, linewidth=0, antialiased=False)
ax.set_zlim(-0.2,0.2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
figure.colorbar(surface)
plt.show()
