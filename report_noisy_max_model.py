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
ax.plot(xs, ys, color="blue", linewidth=2.0, linestyle="-", label="Pr(A)")
ys = [b.pdf(x) for x in xs]
ax.plot(xs, ys, color="green", linewidth=2.0, linestyle="-", label="Pr(B)")
ax.legend(loc='upper right')
plt.ylim(0,0.1)
plt.show()
