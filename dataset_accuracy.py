#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from algorithms import epsilon
from accuracy import probability_precise
from accuracy import probability_optimized
from experiments import *


def plot_single(data, func, e, c, s):
    fig, ax = plt.subplots(figsize=(7,4))
    _counts, array = read_data(data)
    ys = []
    std = []
    results = np.genfromtxt('experiments/{}-{} {} {}.txt'.format(data, func.__name__, c, s), dtype=None)
    ser, prob = to_pdf(results, c, array)
    ax.bar(ser, prob, color="red", align='edge',
           label=r"$\mathrm{Pr}(\mathrm{SER} \leq x) \geq 1 - \beta$")

    ax.bar(ser, discrete_pdf(prob), width=1/100, color="blue", align='edge',
           label=r"$\mathrm{Pr}(\mathrm{SER} = x) \approx 1 - \beta$")

    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$1 - \beta$")
    ax.legend(loc='upper left')
    plt.title(r"{}, $k = {}, c = {}, \epsilon = {}$".format(datasets[data], len(array), c, e))
    plt.show()


if __name__ == '__main__':
    queries = np.loadtxt('data/bms-pos.txt', dtype=int)
    c = 50
    e = 0.1
    ratio = 'c23'
    e1, e2 = epsilon(e, ratios(c)[ratio])

    plot_single('bms-pos', basic, e, c, ratio)
    plot_single('bms-pos', precise, e, c, ratio)
