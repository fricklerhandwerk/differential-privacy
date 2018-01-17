from math import sqrt
from algorithms import *
from matplotlib import pyplot as plt


def lines(scale_a, loc_a, scale_b, loc_b):
    sig_a = 2 * scale_a**2
    sig_b = 2 * scale_b**2
    a_x, a_y = [loc_b, loc_b], [loc_a - sig_a/2, loc_a + sig_a/2]
    b_x, b_y = [loc_a, loc_a], [loc_b - sig_b/2, loc_b + sig_b/2]
    return a_x, a_y, b_x, b_y

a_x, a_y, b_x, b_y = lines(1,2,1.5,4)

fig, ax = plt.subplots()
ax.plot(range(10))
ax.plot(a_x, a_y, 'r-')
ax.plot(a_x, a_y, 'r_')
ax.plot(b_y, b_x, 'r-')
ax.plot(b_y, b_x, 'r|')
plt.xlabel("A")
plt.ylabel("B")
plt.xlim(0,10)
plt.ylim(0,10)
plt.show()
