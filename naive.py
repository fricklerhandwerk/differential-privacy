#!/usr/bin/env python3

from numpy import random


def Lap(scale):
    return random.laplace(scale=scale)


def report_noisy_max(database, queries, epsilon):
    responses = [q(database) + Lap(1/epsilon) for q in queries]
    return responses.index(max(responses))


def above_threshold(database, queries, threshold, e1, e2, sensitivity=1, monotonic=True):
    result = []
    r = Lap(sensitivity/e1)
    for q in queries:
        factor = 1 if monotonic else 2
        v = Lap(factor*sensitivity/e2)
        if q(database) + v >= threshold + r:
            result.append(True)
            break
        else:
            result.append(False)
    return result


def sparse(database, queries, threshold, count, e1, e2, sensitivity=1, monotonic=True):
    result = []
    r = Lap(sensitivity/e1)
    c = 0
    for q in queries:
        factor = 1 if monotonic else 2
        v = Lap(factor*count*sensitivity/e2)
        if q(database) + v >= threshold + r:
            result.append(True)
            c += 1
            if c >= count:
                break
        else:
            result.append(False)
    return result
