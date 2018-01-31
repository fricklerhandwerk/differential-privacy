#!/usr/bin/env python3

import numpy as np
from scipy.integrate import IntegrationWarning
from collections import defaultdict

from accuracy import probability_precise
from accuracy import accuracy_optimized
from algorithms import scale
from algorithms import epsilon
from algorithms import factor
from dataset_accuracy import above
from dataset_accuracy import below
from dataset_accuracy import threshold
from dataset_accuracy import uniq_xs

datasets = ["bms-pos", "kosarak", "aol"]

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


for d in datasets:
    queries = np.loadtxt('data/{}.txt'.format(d), dtype=int)
    k = queries[0]
    for c in cs:
        # compute probabilities only for unique tuples with numbers of queries
        # above and below the T+/-alpha range
        T = threshold(c, queries)
        above_below = {(len(below(queries, T, a)), len(above(queries, T, a))): a
                       for a in range(k)}
        alphas = list(above_below.values())

        for s, r in ratios(c).items():
            print("c: {}, r: {}".format(c, s))
            s1, s2 = scale(*epsilon(e, r), c)
            with open('experiments/{} {} {}.txt'.format(d, c, s), 'a') as f:
                last = 1
                for i, a in enumerate(alphas):
                    p = probability_precise(a, k, s1, s2)
                    # catch problems with integration
                    if p > last or last == 0:
                        break
                    else:
                        last = p
                    print(len(alphas) - i, a, p, end='\r')
                    print(a, p, file=f)
                print()
