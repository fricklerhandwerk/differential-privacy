import gmpy2
from gmpy2 import comb
from gmpy2 import exp
from gmpy2 import fsum
from math import log
from math import isclose
from numpy import nextafter
from scipy.optimize import root

from matplotlib import pyplot as plt

from efprob.dc import *
from algorithms import *

gmpy2.get_context().precision = 1000

def threshold_accuracy(b, e):
    return 2*log(2/b)/e


def queries_accuracy(k, b, e):
    return 4*(log(k) + log(2/b))/e


def accuracy_overestimate(b, k, e1, e2):
    # the definition in [@privacybook, p. 60] is only valid for e1 = e2 = e/2.
    return max(threshold_accuracy(b, e1), queries_accuracy(k, b, e2))


def beta2(b, e1, e2, k):
    def opt(x):
        return b - x - (k/x)**(-2*e1/e2)
    return root(opt, nextafter(b,0), tol=nextafter(0,1)).x[0]


def beta1(b, e1, e2, k):
    def opt(x):
        return b - x - k*(x**(e2/(2*e1)))
    return root(opt, nextafter(b,0), tol=nextafter(0,1)).x[0]


def accuracy_improved(b2, e2, k):
    return 4*log(k/b2)/e2


def accuracy_optimal(b, k, e1, e2):
    a1 = (log(2*e1+e2) - log(e2*b))/e1
    a2 = (2/e2) * (log(k) + log(2*e1+e2) - log(2*e1*b))
    return a1 + a2


def threshold(x, e1):
    return exp(-(x/2)*e1)


def queries(x, k, e2):
    """upper bound on probability that any of k queries is >= x"""
    return clip(k*exp(-(x/2)*e2/2))


def queries_improved(x, k, e2):
    """precise probability that any of k queries is >= x"""
    def f(l):
        return (-1)**l * exp(-l*(x/2)*e2/2)
    result = -fsum(comb(k, l) * f(l) for l in range(1, k+1))
    return clip(result)


def total_overestimate(x, k, e1, e2):
    # we have to take factor two since we assume b1 = b2 = b/2,
    # and each noise factor accounts for only one part of the probability budget
    return 2*max(threshold(x, e1), queries(x, k, e2))


def total(x, k, e1, e2):
    """bound on total noise"""
    # inverse function of accuracy_improved
    return clip(threshold(x, e1) + queries(x, k, e2))


def total_improved(x, k, e1, e2):
    """improved bound of total noise"""
    return clip(threshold(x, e1) + queries_improved(x, k, e2))


def total_optimal(x, k, e1, e2):
    # inverse function of accuracy_optimal
    e = 2*e1+e2
    wow = (e)/(((exp(e1*e2*x)*(e2**e2))/((k/(2*e1))**(2*e1)))**(1/e))
    return clip(wow)


def clip(x):
    return min(1, x)


def plot():
    k = 10
    b = 0.2
    e1 = 0.1
    e2 = 0.2
    e = e1 + e2

    b1 = beta1(b, e1, e2, k)
    b2 = beta2(b, e1, e2, k)

    MAX = max(accuracy_overestimate(b, k, e1, e2), accuracy_improved(b2, e2, k), accuracy_optimal(b, k, e1, e2))

    print(accuracy_overestimate(b, k, e1, e2))
    print(accuracy_improved(b2, e2, k))
    print(accuracy_optimal(b, k, e1, e2))

    assert isclose(total(accuracy_improved(b2, e2, k), k, e1, e2), b), total(accuracy_improved(b2, e2, k), k, e1, e2)

    fig, ax = plt.subplots()
    plt.ylim(0,1)
    plt.xlim(0,MAX)

    xs = np.arange(MAX)
    ax.plot(xs, [total_overestimate(x, k, e1, e2) for x in xs], color="pink", linewidth=2.0, label="overestimate")
    ax.plot(xs, [total(x, k, e1, e2) for x in xs], color="red", linewidth=2.0, label="baseline")
    ax.plot(xs, [total_improved(x, k, e1, e2) for x in xs], color="green", linewidth=2.0, label="improved")
    ax.plot(xs, [total_optimal(x, k, e1, e2) for x in xs], color="blue", linewidth=2.0, label="optimal")
    ax.legend(loc='upper right')
    plt.show(block=False)

    fig, ax = plt.subplots()
    plt.ylim(0,1)
    plt.xlim(0,MAX)
    ys = np.linspace(0.001,1,256)
    ax.plot([accuracy_overestimate(y, k, e1, e2) for y in ys], ys, color="pink", linewidth=2.0, label="overestimate")
    ax.plot([accuracy_improved(beta2(y, e1, e2, k), e2, k) for y in ys], ys, color="red", linewidth=2.0, label="baseline")
    ax.plot([accuracy_optimal(y, k, e1, e2) for y in ys], ys, color="blue", linewidth=2.0, label="optimal")
    ax.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    plot()
