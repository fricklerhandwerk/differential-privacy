#!/usr/bin/env python3

from functools import partial
import warnings

from algorithms import *

warnings.filterwarnings('ignore')

N = 1000  # database size

def create_database(p, q, n):
    return {'a': int(p*n), 'b': int(q*n)}

database = create_database(0.500, 0.500, N)
queries = [lambda x: x['a'], lambda x: x['b']]

result = report_noisy_max(database, queries, epsilon=0.1)

plot(result, interval=R(0,400), steps=512, block=False)
print("P('a' > 'b'):")
print((result[0] @ result[1]) >= a_larger_b())
input("Press [enter] to continue.")
