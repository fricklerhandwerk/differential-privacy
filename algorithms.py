#!/usr/bin/env python3

from efprob.dc import *
from math import exp


R_plus = R(0, inf)
R_minus = R(-inf, 0)

def Lap(b, m=0):
    def laplace(x):
        return 1/(2*b)*exp(-abs(x-m)/b)
    return State.fromfun(laplace, R_plus)

def plot(states, preargs=(), interval=None,
         postargs=(), steps=256, block=True):
    fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(10, 5))
    for s in states:
        axis = len(preargs)
        if interval is None:
            interval = s.dom[axis]
        if interval[1] < interval[0]:
            raise ValueError("Empty interval")
        if math.isinf(interval[0]) or math.isinf(interval[1]):
            raise ValueError("Unbounded interval")
        xs = np.linspace(interval[0], interval[1], steps, endpoint=True)
        ys = [s(*(preargs+(x,)+postargs)) for x in xs]
        ax.plot(xs, ys, color="blue", linewidth=2.0, linestyle="-")
    plt.draw()
    plt.pause(0.001)
    if block:
        input("Press [enter] to continue.")

a_larger_b = Predicate(lambda x, y: 1 if x >= y else 0, [R_plus, R_plus])

def report_noisy_max(database, queries, epsilon):
    return [Lap(1/epsilon, q(database)) for q in queries]
