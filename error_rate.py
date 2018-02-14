import matplotlib.pyplot as plt
from matplotlib import cm as colormap
import numpy as np

from numpy import product
from algorithms import *

epsilon1 = 0.05
epsilon2 = 0.05
k = 1

threshold = Laplace(1/epsilon1, loc=0)

def pr_query_response(is_above, query, threshold):
    """Pr(query => is_above | threshold)"""
    pr_below = Laplace(k/epsilon2, loc=query).cdf(threshold)
    if not is_above:
        return pr_below
    else:
        return 1 - pr_below

def pr_vector_threshold(a, b, q, p):
    """Pr(qs => rs | threshold)"""
    def pred(x):
        return pr_query_response(a, q, x)**k * pr_query_response(b, p, x)**k
    return Predicate(pred, R)

def get_pr_correct(x, y):
    """
    probability that two queries positioned at x, y are answered
    correctly with respect to `threshold`
    """
    pr_correct_response = pr_vector_threshold(x >= 0, y >= 0, x, y)
    """wow, the cool thing is that the probability of getting the exactly correct vector
    for queries *just around* the threshold is minimal"""
    return threshold.state >= pr_correct_response


figure = plt.figure(figsize=(5,3))
ax = figure.gca(projection="3d")

X = np.r_[-30:-11:4, -10:-3:1, -2:2:0.2, 3:10:1, 11:30:4]
Y = np.r_[-30:-11:4, -10:-3:1, -2:2:0.2, 3:10:1, 11:30:4]
print("number of values to compute:", len(X)**2, "- please wait...")
X, Y = np.meshgrid(X, Y)
Z = np.vectorize(get_pr_correct)(X, Y)
ax.plot_surface(X, Y, Z, cmap=colormap.viridis, linewidth=0, antialiased=False)
plt.title(r"$\epsilon$ = {:.1f}, k = {}".format(epsilon1 + epsilon2, k))
ax.set_zlim(0, 0.1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.azim = -10
ax.elev = 45
plt.show()
