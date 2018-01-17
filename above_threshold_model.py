from math import sqrt
import random
import numpy as np
from algorithms import *
from matplotlib import pyplot as plt


epsilon1 = 0.1
epsilon2 = 0.15

LEN = 8
MAX = 300
T = 200
locs = [random.randint(0,MAX) for _ in range(LEN)]
queries = [Laplace(1/epsilon1, i) for i in locs]
threshold = Laplace(1/epsilon2, T)

fig, ax = plt.subplots()
xs = np.arange(MAX)
ys = [threshold.pdf(x) for x in xs]
ax.plot(xs, ys, color="red", linewidth=2.0, linestyle="-", label="Pr(T)")
ax.plot([], color="blue", linewidth=2.0, linestyle="-", label="Pr(q$_i$)")
for q in queries:
	ys = [q.pdf(x) for x in xs]
	ax.plot(xs, ys, color="blue", linewidth=2.0, linestyle="-")
ax.legend(loc='upper right')
plt.ylim(0,0.1)
plt.show()
