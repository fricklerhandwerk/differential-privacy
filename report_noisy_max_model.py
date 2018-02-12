from math import sqrt
import numpy as np
from algorithms import *
import matplotlib as mpl
from matplotlib import pyplot as plt

ticklabelpad = mpl.rcParams['xtick.major.pad']


epsilon = 0.1

a = Laplace(1/epsilon, 180)
b = Laplace(1/epsilon, 150)


fig, ax = plt.subplots(figsize=(5, 2))
xs = np.arange(250)
ys = [a.pdf(x) for x in xs]
ax.fill_between(xs, ys, color="blue", linewidth=0, linestyle="-", label="Pr(A = x)", alpha=0.8)
ys = [b.pdf(x) for x in xs]
ax.fill_between(xs, ys, color="green", linewidth=0, linestyle="-", label="Pr(B = x)", alpha=0.8)
ax.legend(loc='upper right')
ax.annotate('x', xy=(0.5,0), xytext=(0, -2*ticklabelpad), ha='left', va='top', xycoords='axes fraction', textcoords='offset points')
plt.ylim(0,0.1)
plt.show()
