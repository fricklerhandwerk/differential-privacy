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

fig, ax = plt.subplots(figsize=(6,4))
xs = np.arange(MAX)
ax.fill_between([], [], color="blue", linewidth=0, linestyle="-", label="$\mathbb{P}(\hat{q}_i = x)$", alpha=0.8)
for q in queries:
	ys = [q.pdf(x) for x in xs]
	ax.fill_between(xs, ys, color="blue", linewidth=0, linestyle="-", alpha=0.8)
ys = [threshold.pdf(x) for x in xs]
ax.fill_between(xs, ys, color="red", linewidth=0, linestyle="-", label="$\mathbb{P}(\hat{T} = x)$", alpha=0.8)
ax.legend(loc='upper right')
ax.set_xlabel('x')
plt.xlim(0,MAX)
plt.ylim(0,0.1)
plt.show()
