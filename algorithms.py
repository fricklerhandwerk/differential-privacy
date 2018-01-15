#!/usr/bin/env python3

from efprob.dc import *
from math import copysign
from math import erf
from math import exp
from math import pi
from math import sqrt


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


class Laplace(object):
    """Laplace distribution"""
    def __init__(self, scale, mean=0):
        self.scale = scale
        self.mean = mean

    def __call__(self, args):
        return self.state(args)

    @property
    def state(self):
        return Lap(self.scale, self.mean)

    @property
    def cdf(self):
        return LapCDF(self.scale, self.mean)

    def difference(self, other):
        return DiffLap(self.scale, other.scale, self.mean, other.mean)

    def differenceCDF(self, other):
        return DiffLapCDF(self.scale, other.scale, self.mean, other.mean)

    def larger(self, other):
        return 1 - self.differenceCDF(other)(0)


def Lap(b, m=0):
    def laplace(x):
        return exp(-abs(x-m)/b) / (2*b)
    return State.fromfun(laplace, R)


def LapCDF(b, m=0):
    def laplaceCDF(x):
        t = abs(x - m)
        s = sgn(x-m)
        return (-s * exp(-t/b) + s + 1) / 2
    return State.fromfun(laplaceCDF, R)


def DiffLap(a, b, m=0, n=0):
    """difference of two Laplace distributions"""
    def difference(x):
        t = abs(x+n-m)
        k = exp(-t/a)
        l = exp(-t/b)
        if a == b:
            return (k + t/a*k) / (4*a)
        else:
            return ((k+l)/(a+b) + (k-l)/(a-b)) / 4
    return State.fromfun(difference, R)


def DiffLapCDF(a, b, m=0, n=0):
    """CDF of difference of two Laplace distributions"""
    def differenceCDF(x):
        t = abs(x+n-m)
        s = sgn(x+n-m)
        k = exp(-t/a)
        l = exp(-t/b)
        if a == b:
            return (-s * (2*a+t)*k / (2*a) + 1 + s)/2
        else:
            return (-s * ((a*k + b*l)/(a+b) + (a*k - b*l)/(a-b)) / 2 + 1 + s)/2
    return State.fromfun(differenceCDF, R)


class Gauss(object):
    """Gaussian distribution"""
    def __init__(self, scale, mean=0):
        self.scale = scale
        self.mean = mean

    def __call__(self, args):
        return self.state(args)

    @property
    def state(self):
        return Normal(self.scale, self.mean)

    @property
    def cdf(self):
        return NormalCDF(self.scale, self.mean)

    def difference(self, other):
        return DiffNormal(self.scale, other.scale, self.mean, other.mean)

    def differenceCDF(self, other):
        return DiffNormalCDF(self.scale, other.scale, self.mean, other.mean)

    def larger(self, other):
        return 1 - self.differenceCDF(other)(0)


def Normal(b, m=0):
    def gauss(x):
        return exp((-(x-m)**2)/(2*b**2))/sqrt(2*pi*b**2)
    return State.fromfun(gauss, R)


def NormalCDF(b, m=0):
    def gaussCDF(x):
        return (1 + erf((x-m) / (b*sqrt(2)))) / 2
    return State.fromfun(gaussCDF, R)


def DiffNormal(a, b, m=0, n=0):
    def difference(x):
        u = m - n
        s = a**2 + b**2
        return exp((-(x-u)**2)/(2*s**2))/sqrt(2*pi * s)
    return State.fromfun(difference, R)


def DiffNormalCDF(a, b, m=0, n=0):
    def differenceCDF(x):
        u = m - n
        s = a**2 + b**2
        return (1 + erf((x-u) / (s*sqrt(2)))) / 2
    return State.fromfun(differenceCDF, R)


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
