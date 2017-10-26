#!/usr/bin/env python3

from numpy import random


def Lap(scale):
    return random.laplace(scale=scale)


def report_noisy_max(database, queries, epsilon):
    responses = [q(database) + Lap(1/epsilon) for q in queries]
    return responses.index(max(responses))


def above_threshold(database, queries, threshold, e1, e2, sensitivity=1):
    result = []
    r = Lap(sensitivity/e1)
    for q in queries:
        v = Lap(2*sensitivity/e2)
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


def numeric_sparse(database, queries, threshold, count, e1, e2, e3=0, sensitivity=1, monotonic=True):
    """
    e1, e2, e3 are privacy budgets for the threshold, the query and the numeric
    results respectively.
    """
    result = []
    r = Lap(sensitivity/e1)
    c = 0
    for q in queries:
        factor = 1 if monotonic else 2
        v = Lap(factor*count*sensitivity/e2)
        if q(database) + v >= threshold + r:
            if e3 > 0:
                result.append(q(database) + Lap(c*sensitivity/e3))
            else:
                result.append(True)
            c += 1
            if c >= count:
                break
        else:
            result.append(False)
    return result


"""
handling arguments needs to be more elaborate if we want to have a fallback
for generating threshold noise locally.
"""


def above_threshold_2(database, queries, threshold, e, r, sensitivity=1, monotonic=True):
    """note that we pass `r` so we don't need to reset it on iteration."""
    result = []
    for q in queries:
        factor = 1 if monotonic else 2
        v = Lap(factor*sensitivity/e)
        if q(database) + v >= threshold + r:
            result.append(True)
            break
        else:
            result.append(False)
    return result


def sparse_2(database, queries, threshold, count, e, r, sensitivity=1, monotonic=True):
    result = []
    for c in range(count):
        # exhaust queries on each run
        q = queries[len(result):]
        # note that we pass increased noise with sensitivity implicitly
        partial = above_threshold_2(database, q, threshold, e, r, count*sensitivity, monotonic)
        result.extend(partial)
    return result


def numeric_sparse_2(database, queries, threshold, count, e1, e2, e3=0, sensitivity=1, monotonic=True):
    r = Lap(sensitivity/e1)
    vector = sparse_2(database, queries, threshold, count, e2, r, sensitivity, monotonic)
    if e3 > 0:
        return [q(database) + Lap(count*sensitivity/e3) if t else t
                for t, q in zip(vector, queries)]
    else:
        return vector
