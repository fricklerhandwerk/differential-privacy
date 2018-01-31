from math import log
from math import isclose
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root

from matplotlib import pyplot as plt

from efprob.dc import *
from algorithms import *


def accuracy_threshold(b1, s1):
    # we use factor two because a1 = a2 = a/2
    return -2*s1*log(b1)


def accuracy_queries(b2, k, s2):
    # we use factor two because a1 = a2 = a/2
    return -2*s2*(log(b2/k))


def accuracy_queries_improved(b2, k, s2):
    return -2*s2*log(1 - (1 - b2)**(1/k))


def accuracy_overestimate(b, k, s1, s2):
    # the definition in [@privacybook, p. 60] is only valid for
    # e1 = e2 = e/2, sensitivity = 1, monotonic = False.
    # here we only assume a1 = a2 = a/2 and b1 = b2 = b/2
    return max(accuracy_threshold(b/2, s1), accuracy_queries(b/2, k, s2))


def accuracy_baseline(b, k, s1, s2):
    b2 = beta2_baseline(b, k, s1, s2)
    queries = accuracy_queries(b2, k, s2)
    return queries


def beta1_baseline(b, k, s1, s2):
    def opt(b1):
        return b - b1 - k*(b1**(s1/s2))
    return optimize(opt, b/2)


def beta2_baseline(b, k, s1, s2):
    def opt(b2):
        return b - b2 - (b2/k)**(s2/s1)
    return optimize(opt, b/2)


def accuracy_improved(b, k, s1, s2):
    b2 = beta2_improved(b, k, s1, s2)
    queries = accuracy_queries_improved(b2, k, s2)
    return queries


def beta1_improved(b, k, s1, s2):
    def opt(b1):
        return b - b1 - 1 + (1 - b1**(s1/s2))**k
    return optimize(opt, b/2)


def beta2_improved(b, k, s1, s2):
    def opt(b2):
        return b - b2 - (1 - (1 - b2)**(1/k))**(s2/s1)
    return optimize(opt, b/2)


def accuracy_optimized(b, k, s1, s2):
    return s1 * log((s2/s1 + 1)/b) + s2 * log(k*(s1/s2 + 1)/b)


def optimize(func, guess):
    return root(func, guess).x[0]


def threshold(a1, s1):
    return exp(-a1/s1)


def queries(a2, k, s2):
    """upper bound on probability that any of k queries is >= x"""
    return clip(k*exp(-a2/s2))


def queries_improved(a2, k, s2):
    """precise probability that any of k queries is >= x"""
    return clip(1 - (1 - exp(-a2/s2))**k)


def probability_overestimate(a, k, e1, e2):
    # we have to take factor two on the resulting probability since we assume b1 = b2 = b/2,
    # and each noise factor accounts for only one part of the probability budget
    # same goes for the argument `a`, where we assume a1 = a2 = a/2
    return clip(2 * max(threshold(a/2, e1), queries(a/2, k, e2)))


def probability_baseline(a, k, s1, s2):
    """bound on total noise"""
    # inverse function of accuracy_improved
    # allows b1 != b2, but still assumes a1 = a2 = a/2
    return clip(threshold(a/2, s1) + queries(a/2, k, s2))


def probability_improved(a, k, s1, s2):
    """improved bound of total noise"""
    # allows b1 != b2, but still assumes a1 = a2 = a/2
    return clip(threshold(a/2, s1) + queries_improved(a/2, k, s2))


def probability_optimized(a, k, s1, s2):
    # inverse function of accuracy_optimized
    return clip((((s2/s1 + 1)**(s1/(s1 + s2)) * (k*(s1/s2 + 1))**(s2/(s1 + s2))) / exp(a/(s1 + s2))))


def probability_precise(x, k, s1, s2):
    def inner(z):
        def wrap(t):
            return exp((t-z)/s1 - t/s2) * ((1-exp(-t/s2))**(k-1))
        return wrap
    def outer(z):
        return (k/(s1*s2)) * quad(inner(z), 0, z)[0]
    return 1 - quad(outer, 0, x)[0]


def clip(x):
    return min(1, x)


def plot():
    e = 0.3
    e1, e2 = epsilon(e, ratio=2)
    s1, s2 = scale(e1, e2, c=1, sensitivity=1, monotonic=False)

    k = 10
    b = 0.2

    example = [accuracy_overestimate(b, k, s1, s2), accuracy_baseline(b, k, s1, s2), accuracy_optimized(b, k, s1, s2)]
    MAX = max(example)

    fig, ax = plt.subplots()
    plt.ylim(0,1)
    plt.xlim(0,MAX)

    xs = np.arange(MAX)

    ax.plot(xs, [probability_overestimate(x, k, s1, s2) for x in xs], color="pink", linewidth=2.0, label="overestimate")
    ax.plot(xs, [probability_baseline(x, k, s1, s2) for x in xs], color="red", linewidth=2.0, label="baseline")
    ax.plot(xs, [probability_improved(x, k, s1, s2) for x in xs], color="green", linewidth=2.0, label="improved")
    ax.plot(xs, [probability_optimized(x, k, s1, s2) for x in xs], color="blue", linewidth=2.0, label="optimized")
    ax.plot(xs, [probability_precise(x, k, s1, s2) for x in xs], color="black", linewidth=2.0, label="precise")
    ax.legend(loc='upper right')
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    plt.show(block=False)

    fig, ax = plt.subplots()
    plt.ylim(0,1)
    plt.xlim(0,MAX)
    ys = np.linspace(0.001,1,256)
    ax.plot([accuracy_overestimate(y, k, s1, s2) for y in ys], ys, color="pink", linewidth=2.0, label="overestimate")
    ax.plot([accuracy_baseline(y, k, s1, s2) for y in ys], ys, color="red", linewidth=2.0, label="baseline")
    ax.plot([accuracy_improved(y, k, s1, s2) for y in ys], ys, color="green", linewidth=2.0, label="improved")
    ax.plot([accuracy_optimized(y, k, s1, s2) for y in ys], ys, color="blue", linewidth=2.0, label="optimized")
    ax.legend(loc='upper right')
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    plt.show()

if __name__ == '__main__':
    plot()
