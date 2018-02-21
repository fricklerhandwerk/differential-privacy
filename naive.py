#!/usr/bin/env python3

from numpy import random
from algorithms import epsilon
from algorithms import scale


def Lap(scale):
    return random.laplace(scale=scale)


def report_noisy_max(database, queries, epsilon):
    responses = [database[q] + Lap(1/epsilon) for q in queries]
    return responses.index(max(responses))


def sparse(database, queries, threshold, epsilon, ratio,
           c=1, sensitivity=1, monotonic=True):
    e1, e2 = epsilon(epsilon, ratio)
    s1, s2 = scale(e1, e2, c, sensitivity, monotonic)

    result = []
    r = Lap(s1)
    count = 0

    for q in queries:
        n = Lap(s2)
        if database[q] + n >= threshold + r:
            result.append(True)
            count += 1
            if count >= c:
                break
        else:
            result.append(False)
    return result
