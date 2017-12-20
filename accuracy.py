import gmpy2
from gmpy2 import comb
from gmpy2 import exp
from gmpy2 import fsum
from math import log

from efprob.dc import *
from algorithms import *

gmpy2.get_context().precision = 1000


def accuracy(k, b, e):
    return 8*(log(k) + log(2/b))/e


def reduced_accuracy(b, e):
    return 4*log(2/b)/e

k = 10
b = 0.1
e1 = 0.1
e2 = 0.1
e = e1 + e2
T = 0
a_q = accuracy(k, b, e)
a_T = reduced_accuracy(b, e)


def threshold(x):
    return exp(-x*e1)


def queries(x):
    """upper bound on probability that any of k queries is >= x"""
    return min(1, k*exp(-x*e2/2))


def queries_improved(x):
    """precise probability that any of k queries is >= x"""
    def f(l):
        return (-1)**l * exp(-l*x*e2/2)
    result = -fsum(comb(k, l) * f(l) for l in range(1, k+1))
    return min(1, result)


def total(x):
    """bound on total noise"""
    return min(1, threshold(x) + queries(x))


def total_improved(x):
    """improved bound of total noise"""
    return min(1, threshold(x) + queries_improved(x))


print("alpha_q/2:", a_q/2)
print("alpha_T/2:", a_T/2)
print("b/2:", b/2)
State.fromfun(threshold, R).plot(R(0, a_q))
State.fromfun(queries, R).plot(R(0, a_q))
State.fromfun(queries_improved, R).plot(R(0, a_q))
State.fromfun(total, R).plot(R(0, a_q))
State.fromfun(total_improved, R).plot(R(0, a_q))
