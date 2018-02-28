#!/usr/bin/env python3

from math import copysign
from math import exp
from math import floor
from scipy.stats import laplace
from scipy.stats import norm


def epsilon(e, ratio=1):
    e1 = e/(1+ratio)
    e2 = e - e1
    return e1, e2


def factor(monotonic):
    return 1 if monotonic else 2


def scale(e1, e2, c=1, sensitivity=1, monotonic=True):
    m = factor(monotonic)
    s1 = sensitivity / e1
    s2 = sensitivity * m * c / e2
    return s1, s2


def report_noisy_max(database, queries, epsilon):
    return [Laplace(1/epsilon, database[q]) for q in queries]


def sparse(database, queries, threshold, epsilon, ratio,
           c=1, sensitivity=1, monotonic=True):
    """
    since this is an abstraction to random distributions, and we have no
    obvious way to halt after some condition is met, it is
    the caller's obligation to sample from the distributions and cut off
    the result when `c` positive answers are collected.
    """
    e1, e2 = epsilon(epsilon, ratio)
    s1, s2 = scale(e1, e2, c, sensitivity, monotonic)
    T = Laplace(s1, threshold)
    return T, [Laplace(s2, database[q]) for q in queries]


def exponential(database, utility, epsilon, sensitivity=1, monotonic=True):
    m = factor(monotonic)

    def weight(x):
        return exp((epsilon*utility(database[x]))/(m*sensitivity))

    normalization = sum(weight(x) for x in database)

    def distribution(x):
        return weight(x) / normalization

    return disribution


class Distribution(object):
    def __init__(self, scale, loc=0):
        self.scale = scale
        self.loc = loc

    def __call__(self, x):
        return self.pdf(x)

    def larger(self, other):
        return 1 - self.differenceCDF(other)(0)


class Laplace(Distribution):
    """Laplace distribution"""

    def pdf(self, x):
        return laplace.pdf(x, scale=self.scale, loc=self.loc)

    def cdf(self, x):
        return laplace.cdf(x, scale=self.scale, loc=self.loc)

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
                return (k + (t/a)*k) / (4*a)
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


class Gaussian(Distribution):
    """Gaussian distribution"""

    def pdf(self, x):
        return norm.pdf(x, scale=self.scale, loc=self.loc)

    def cdf(self, x):
        return norm.cdf(x, scale=self.scale, loc=self.loc)

    def difference(self, other):
        def diff(x):
            b = self.scale**2 + other.scale**2
            m = self.loc - other.loc
            return norm.pdf(x, scale=b, loc=m)

        return diff

    def differenceCDF(self, other):
        def diffCDF(x):
            b = self.scale**2 + other.scale**2
            m = self.loc - other.loc
            return norm.cdf(x, scale=b, loc=m)

        return diffCDF


def sgn(x):
    return copysign(1, x)
