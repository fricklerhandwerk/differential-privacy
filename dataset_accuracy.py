#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
from numpy import log
from accuracy import accuracy_optimal


def error_rate(b, k, e1, e2, queries, T, c):
    alpha = accuracy_optimal(b, k, e1, e2)
    worst_scores = np.sum(queries[queries >= (T - alpha)][-c:])
    top_scores = np.sum(queries[:c])
    return 1 - worst_scores/top_scores

if __name__ == '__main__':
    items = np.loadtxt('data/kosarak_clean.json', dtype=int)
    monotonic = True
    sensitivity = 1
    factor = 1 if monotonic else 2
    c = 100
    T = (items[c-1] + items[c])/2
    k = len(items)

    e = 0.1
    e1 = e/(1+(factor*c)**(2/3))
    e2 = e - e1

    fig, ax = plt.subplots()
    plt.xlim(0,1)
    plt.ylim(0,1)
    ys = np.linspace(0.001, 0.999, k/2)
    error_rate_cdf = [error_rate(y, k, e1, e2, items, T, c) for y in ys]
    ax.plot(error_rate_cdf, ys[::-1], color="blue", linewidth=2.0, label="CDF")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$1 - \beta$")
    plt.title(r"$\mathrm{Pr}(\mathrm{SER} \leq x) \leq 1 - \beta$")
    plt.show()