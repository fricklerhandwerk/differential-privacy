#!/usr/bin/env python3

from functools import partial
from random import shuffle

from naive import above_threshold
from naive import report_noisy_max

N = 100  # database size
K = 100  # number of trials


def create_database(p, q, n):
    return {'a': int(p*n), 'b': int(q*n)}

database = create_database(0.2, 0.5, N)
queries = [lambda x: x['a'], lambda x: x['b']]

result = []
for i in range(K):
    result.append(report_noisy_max(database, queries, epsilon=0.1))

print("#max:", result.count(0), "value:", database['a'])
print("#max:", result.count(1), "value:", database['b'])


def create_ranks(n):
    """generate a database of size `n` according to Zipf's law"""
    return {i: n/(i+1) for i in range(n)}


def query(i, x):
    return x[i]

ranks_database = create_ranks(N)
ranks_queries = [partial(query, i) for i in range(N)]


print("Average length of response vector depending on threshold")
for m in range(K//4):
    shuffle(ranks_queries)
    T = N/(m+1)
    def result():
        result = above_threshold(ranks_database, ranks_queries, T, e1=0.1, e2=0.1)
        assert not any(result[:-1])
        return result
    print("T:", "{:.2f}".format(T), "len:", sum(len(result()) for _ in range(K))/K)

