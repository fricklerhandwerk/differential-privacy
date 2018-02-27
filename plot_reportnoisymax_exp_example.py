from math import exp
from math import sqrt
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

ticklabelpad = mpl.rcParams['xtick.major.pad']

epsilon = 0.1
x = 150
y = 180
norm = 1 / sum(exp(epsilon * i) for i in (x, y))

a = norm * exp(epsilon * x)
b = norm * exp(epsilon * y)


fig, ax = plt.subplots(figsize=(5, 2))
# ax.bar((x, y), (a,b), color="orange", width=10, label=r"$\mathbb{P}$(q(x) is max.)", zorder=0)
xs = np.arange(x, 250)
ys = [epsilon*exp(-epsilon*(i-x)) for i in xs]
ax.fill_between(xs, ys, color="blue", linewidth=0, linestyle="-", label="Pr(A = x)", alpha=0.8, zorder=2)
xs = np.arange(y, 250)
ys = [epsilon*exp(-epsilon*(i-y)) for i in xs]
ax.fill_between(xs, ys, color="green", linewidth=0, linestyle="-", label="Pr(B = x)", alpha=0.8, zorder=1)
ax.annotate('x', xy=(0.5,0), xytext=(0, -2*ticklabelpad), ha='left', va='top', xycoords='axes fraction', textcoords='offset points')
ax.legend(loc='upper left')
plt.xlim(0,250)
plt.ylim(0,0.1)
plt.show()
