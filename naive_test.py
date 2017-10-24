#!/usr/bin/env python3

from naive import report_noisy_max


def create_database(p, q, n):
    return {'a': int(p*n), 'b': int(q*n)}

database = create_database(0.2, 0.5, 100)
queries = [lambda x: x['a'], lambda x: x['b']]

result = []
for i in range(100):
    result.append(report_noisy_max(database, queries, epsilon=0.05))

print("#max:", result.count(0), "value:", database['a'])
print("#max:", result.count(1), "value:", database['b'])


