from pprint import pprint as pp

import gmpy2
from gmpy2 import comb, fsum, mpfr, exp
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
a = accuracy(k, b, e)
a_ = reduced_accuracy(b, e)

threshold = lambda x: exp(-x/(1/e1))
queries = lambda x: min(1, k*exp(-x/(2/e2)))


def the_real_shit(x):
    """Probability that any of k queries is >= x"""
    def dude(l):
        return (-gmpy2.exp(-x/(2/e2)))**l
    result = -fsum(comb(k, l) * dude(l) for l in range(1, k+1))
    # if abs(result) > 1.1:
    #     return 0
    return result


def totalfailure(x):
    """Union bound on total noise"""
    return min(1, threshold(x) + queries(x))


def absolutefailure(x):
    """precise calculation of total noise"""
    return min(1, threshold(x) + the_real_shit(x))

shit = State.fromfun(threshold, R)
shit2 = State.fromfun(queries, R)
shit2a = State.fromfun(the_real_shit, R)
shit3 = State.fromfun(totalfailure, R)
shit4 = State.fromfun(absolutefailure, R)
print("alpha/2:", a/2)
print("alpha_/2:", a_/2)
print("beta/2:", b/2)
shit.plot(R(00, a+10))
shit2.plot(R(00, a+10))
shit2a.plot(R(00, a+10))
shit3.plot(R(00, a+10))
shit4.plot(R(00, a+10))
