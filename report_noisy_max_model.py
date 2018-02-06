from math import sqrt
import numpy as np
from algorithms import *
from matplotlib import pyplot as plt


epsilon = 0.1

a = Laplace(1/epsilon, 180)
b = Laplace(1/epsilon, 150)


fig, ax = plt.subplots()
xs = np.arange(250)
ys = [a.pdf(x) for x in xs]
ax.fill_between(xs, ys, color="blue", linewidth=0, linestyle="-", label="Pr(A)", alpha=0.8)
ys = [b.pdf(x) for x in xs]
ax.fill_between(xs, ys, color="green", linewidth=0, linestyle="-", label="Pr(B)", alpha=0.8)
ax.legend(loc='upper right')
plt.ylim(0,0.1)
plt.show()
