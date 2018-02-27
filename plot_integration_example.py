from math import sqrt
from algorithms import *
from matplotlib import pyplot as plt


def lines(scale_a, loc_a, scale_b, loc_b):
    var_a = 2 * scale_a**2
    var_b = 2 * scale_b**2
    a_x, a_y = [loc_b, loc_b], [loc_a - var_a/2, loc_a + var_a/2]
    b_x, b_y = [loc_a, loc_a], [loc_b - var_b/2, loc_b + var_b/2]
    return a_x, a_y, b_x, b_y

a_x, a_y, b_x, b_y = lines(1, 2, 1.5, 4)

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(range(11))
ax.plot(a_y, a_x, 'r-')
ax.plot(a_y, a_x, 'r|')
ax.plot(b_x, b_y, 'r-')
ax.plot(b_x, b_y, 'r_')
plt.xlabel("dom(X)")
plt.ylabel("dom(Y)")
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.show()
