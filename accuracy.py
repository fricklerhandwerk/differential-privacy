import gmpy2
from gmpy2 import comb
from gmpy2 import exp
from gmpy2 import fsum
from math import log
from math import isclose

from efprob.dc import *
from algorithms import *

gmpy2.get_context().precision = 1000


def threshold_accuracy(b, e):
    return 4*log(2/b)/e


def accuracy(k, b, e):
    return 8*(log(k) + log(2/b))/e


def real_accuracy(k, b, e1, e2):
    # this is only valid for fixed ratio of e1:e2, because we have to find b1 first.
    # b1 = (k/b2)**(-2*e1/e2)
    assert isclose(2*e1, e2)
    b1 = 2/110
    b2 = 2/11
    return -2*log(b1)/e1

k = 10
b = 0.2
e1 = 0.1
e2 = 0.2
e = e1 + e2
T = 0
a_q = accuracy(k, b, e)
a_T = threshold_accuracy(b, e)


def threshold(x):
    return exp(-(x/2)*e1)


def queries(x):
    """upper bound on probability that any of k queries is >= x"""
    return min(1, k*exp(-(x/2)*e2/2))


def queries_improved(x):
    """precise probability that any of k queries is >= x"""
    def f(l):
        return (-1)**l * exp(-l*(x/2)*e2/2)
    result = -fsum(comb(k, l) * f(l) for l in range(1, k+1))
    return min(1, result)


def total(x):
    """bound on total noise"""
    return min(1, threshold(x) + queries(x))


def total_improved(x):
    """improved bound of total noise"""
    return min(1, threshold(x) + queries_improved(x))


def accuracy_optimal(b):
    a1 = (log(2*e1+e2) - log(e2*b))/e1
    a2 = (2/e2) * (log(k) + log(2*e1+e2) - log(2*e1*b))
    return a1 + a2


def total_alpha_improved(x):
    # inverse function of accuracy_optimal
    e = 2*e1+e2
    wow = (e)/(((exp(e1*e2*x)*(e2**e2))/((k/(2*e1))**(2*e1)))**(1/e))
    return min(1,wow)


MAX = 150
# State.fromfun(threshold, R).plot(R(0, 200))
# State.fromfun(queries, R).plot(R(0, 200))
# State.fromfun(queries_improved, R).plot(R(0, a_q))
print(accuracy(k,b,e))
print(real_accuracy(k,b,e1, e2))
print(total(real_accuracy(k,b,e1, e2)))
print(accuracy_optimal(b))
State.fromfun(total, R).plot(R(0, MAX))
State.fromfun(total_alpha_improved, R).plot(R(0, MAX))
State.fromfun(total_improved, R).plot(R(0, MAX))
