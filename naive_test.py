#!/usr/bin/env python3

from functools import partial
from random import shuffle

from naive import above_threshold
from naive import report_noisy_max
from naive import sparse

N = 1000  # database size
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


# print("Average length of response vector depending on threshold")
# for m in range(20):
#     T = N/(m+1)
#     def result():
#         shuffle(ranks_queries)
#         result = above_threshold(ranks_database, ranks_queries, T, e1=0.1, e2=0.1)
#         assert not any(result[:-1])
#         return result
#     print("T:", "{:.2f}".format(T), "len:", sum(len(result()) for _ in range(K))/K)


print("Average length of response vector for c above-threshold query results")
for c in range(1,50):
    ranks_queries = [partial(query, i) for i in range(N)]
    T = (ranks_database[c] + ranks_database[c+1])/2
    def result():
        shuffle(ranks_queries)
        return sparse(ranks_database, ranks_queries, T, c, e1=0.1, e2=0.1)
    print("c:", c, "len:", sum(len(result()) for _ in range(K))/K)
