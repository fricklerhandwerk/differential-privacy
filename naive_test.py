#!/usr/bin/env python3

from naive import report_noisy_max


def create_database(p, q, n):
    A = [True] * int(p*n) + [False] * int((100-p)*n)
    B = [True] * int(q*n) + [False] * int((100-q)*n)
    database = {i: {'a': a, 'b': b} for i, (a, b) in enumerate(zip(A, B))}
    return database


def q_a(x):
    return sum(v['a'] for v in x.values())


def q_b(x):
    return sum(v['b'] for v in x.values())


database = create_database(0.2, 0.5, 100)

result = []
for i in range(100):
    result.append(report_noisy_max(database, [q_a, q_b], epsilon=0.05))

print("#max:", result.count('q_a'), "value:", q_a(database))
print("#max:", result.count('q_b'), "value:", q_b(database))
