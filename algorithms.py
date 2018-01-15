#!/usr/bin/env python3

from efprob.dc import *
from math import copysign
from math import erf
from math import exp
from math import pi
from math import sqrt
from scipy.stats import laplace
from scipy.stats import norm


R_plus = R(0, inf)
R_minus = R(-inf, 0)


def report_noisy_max(database, queries, epsilon):
    return [Laplace(1/epsilon, q(database)) for q in queries]


def above_threshold(database, queries, threshold, e1, e2, sensitivity=1, monotonic=True):
    result = []
    T = Laplace(sensitivity/e1, threshold)
    for q in queries:
        factor = 1 if monotonic else 2
        v = Laplace(factor*sensitivity/e2, q(database))
        result.append(flip(v.larger(T)))
    return result


def exponential(database, utility, epsilon, sensitivity=1, monotonic=True):
    factor = 1 if monotonic else 2

    def weight(x):
        return exp((epsilon*utility(database[x]))/(factor*sensitivity))

    normalization = sum(weight(x) for x in database.keys())

    def distribution(x):
        return weight(x)/normalization

    return State.fromfun(distribution, dom=list(database.keys()))


class Distribution(object):
    def __init__(self, scale, loc=0):
        self.scale = scale
        self.loc = loc

    @property
    def state(self):
        return State.fromfun(self.pdf, R)

class Laplace(Distribution):
    """Laplace distribution"""

    def pdf(self, x):
        b = self.scale
        m = self.loc
        return exp(-abs(x-m)/b) / (2*b)

    def cdf(self, x):
        b = self.scale
        m = self.loc
        t = abs(x - m)
        s = sgn(x - m)
        return (-s * exp(-t/b) + s + 1) / 2

    def difference(self, other):
        """difference of two Laplace distributions"""
        a = self.scale
        b = other.scale
        m = self.loc
        n = other.loc

        def diff(x):
            t = abs(x+n-m)
            k = exp(-t/a)
            l = exp(-t/b)
            if a == b:
                return (k + t/a*k) / (4*a)
            else:
                return ((k+l)/(a+b) + (k-l)/(a-b)) / 4

        return diff

    def differenceCDF(self, other):
        a = self.scale
        b = other.scale
        m = self.loc
        n = other.loc

        def diffCDF(x):
            t = abs(x+n-m)
            s = sgn(x+n-m)
            k = exp(-t/a)
            l = exp(-t/b)
            if a == b:
                return (-s * (2*a+t)*k / (2*a) + 1 + s)/2
            else:
                return (-s * ((a*k + b*l)/(a+b) + (a*k - b*l)/(a-b)) / 2 + 1 + s)/2

        return diffCDF

    def larger(self, other):
        return 1 - self.differenceCDF(other)(0)


class Gauss(Distribution):
    """Gaussian distribution"""

    def pdf(self, x):
        return self.normalPDF(x, self.scale, self.loc)

    def cdf(self):
        return self.normalCDF(x, self.scale, self.loc)

    def difference(self, other):
        def diff(x):
            b = self.scale**2 + other.scale**2
            m = self.loc - other.loc
            return self.normalPDF(x, b, m)

        return diff

    def differenceCDF(self, other):
        def diffCDF(x):
            b = self.scale**2 + other.scale**2
            m = self.loc - other.loc
            return self.normalCDF(x, b, m)

        return diffCDF

    def larger(self, other):
        return 1 - self.differenceCDF(other)(0)

    def normalPDF(self, x, b, m):
        return exp((-(x-m)**2)/(2*b**2))/sqrt(2*pi*b**2)

    def normalCDF(self, x, b, m):
        return (1 + erf((x-m) / (b*sqrt(2)))) / 2


def sgn(x):
    return copysign(1, x)


def plot(states, preargs=(), interval=None,
         postargs=(), steps=512, block=True, title=None):
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
    if title:
        plt.title(title)
    plt.draw()
    plt.pause(0.001)
    if block:
        input("Press [enter] to continue.")


def a_larger_b(alpha=0):
    return Predicate(lambda x, y: 1 if x - alpha >= y else 0, [R_plus, R_plus])
