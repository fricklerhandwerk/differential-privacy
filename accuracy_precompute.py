#!/usr/bin/env python3

from collections import Counter
import json
import numpy as np
from scipy.integrate import IntegrationWarning

from accuracy import probability_precise
from accuracy import accuracy_optimized
from algorithms import scale
from algorithms import epsilon
from algorithms import factor
from dataset_accuracy import above
from dataset_accuracy import below
from dataset_accuracy import threshold


datasets = ["bms-pos", "kosarak", "aol", "zipf"]

e = 0.1  # this is what [@svt] used
cs = list(range(1, 50)) + list(range(50, 301, 25))

b_min = 0.01  # only compute probabilities down to this value

def ratios(c, monotonic=True):
    m = factor(monotonic)
    return {
        '1': 1,
        '3': 3,
        'c': m*c,
        'c23': (m*c)**(2/3),
    }


def write_alphas(data, start=1):
    # compute probabilities only for unique tuples with numbers of queries
    # above and below the T+/-alpha range
    queries = np.loadtxt('data/{}.txt'.format(data), dtype=int)
    k = queries[0]
    for c in cs[start:]:
        T = threshold(c, queries)

        above_below = {}
        for a in range(k):
            queries_below = below(queries, T, a)
            queries_above = above(queries, T, a)
            above_below[(len(queries_below), len(queries_above))] = {
                'alpha': a,
                'below': queries_below,
                'above': queries_above,
            }

        print(c, len(above_below), end='\r')

        result = {}
        for _, v in above_below.items():
            result[v['alpha']] = {
                'below': dict(Counter(v['below'].tolist())),
                'above': dict(Counter(v['above'].tolist())),
            }

        with open('experiments/{}-alphas {}.txt'.format(data, c), 'w') as f:
            json.dump(result, f, separators=(',', ':'))


def read_alphas(data, c):
    with open('experiments/{}-alphas {}.txt'.format(data, c)) as f:
        return json.load(f, object_hook=convert)


def convert(d):
    result = {}
    for k, v in d.items():
        if k.isdigit():
            result[int(k)] = v
        else:
            result[k] = v
    return result


def probability_basic(a, k, s1, s2, queries, alphas):
    return probability_precise(a, k, s1, s2)


def write_probability(data, func, start=1):
    queries = np.loadtxt('data/{}.txt'.format(data), dtype=int)
    k = queries[0]
    for c in cs[start:]:
        alphas = read_alphas(data, c).keys()
        total = len(alphas)

        for s, r in ratios(c).items():
            print("c: {}, r: {}".format(c, s))
            s1, s2 = scale(*epsilon(e, r), c)
            with open('experiments/{} {} {}.txt'.format(data, c, s), 'a') as f:
                last = 1
                for i, a in enumerate(alphas):
                    p = func(a, k, s1, s2, queries, alphas)
                    # catch problems with integration
                    if p > last or last == 0:
                        break
                    else:
                        last = p
                    print(total - i, a, p, end='\r')
                    print(a, p, file=f)
                print()
