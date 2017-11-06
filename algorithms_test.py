#!/usr/bin/env python3

from functools import partial
import warnings

from algorithms import *

warnings.filterwarnings('ignore')

N = 50  # database size
K = 100  # number of trials


def create_database(p, q, n):
    return {'a': int(p*n), 'b': int(q*n)}

database = create_database(0.6, 0.2, N)
queries = [lambda x: x['a'], lambda x: x['b']]

result = report_noisy_max(database, queries, epsilon=0.1)

plot(result, interval=R(0,120), steps=512, block=False)
print("P('a' > 'b'):")
print((result[0] @ result[1]) >= a_larger_b())
input("Press [enter] to continue.")


def create_ranks(n):
    """generate a database of size `n` according to Zipf's law"""
    return {i: n/(i+1) for i in range(n)}

ranks_database = create_ranks(N)

e = exponential(ranks_database, lambda x: x, epsilon=0.1)

e.plot()