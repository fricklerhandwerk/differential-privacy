#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from abovethreshold import Model
from accuracy import probability_precise
from accuracy import probability_optimized
from algorithms import *


def threshold(c, queries):
    return (queries[c-1] + queries[c])/2


def error_rate_cdf(error_rate_func, model, step=1):
    xs = (((len(model.below(a)), len(model.above(a))), a) for a in np.arange(model.threshold+1)[::step])
    uniq = list(uniq_xs(xs))
    error_rates = (error_rate_func(model, a) for _, a in uniq)
    return zip(*error_rates)


def error_rate(model, a):
    s1 = model.threshold_scale
    s2 = model.query_scale
    c = model.count
    beta = probability_precise(a, len(model.queries), s1, s2)
    worst_scores = sum(model.queries[model.queries >= model.threshold - a][-c:])
    top_scores = sum(model.queries[:c])
    result = 1 - worst_scores/top_scores, 1 - beta
    return result


def error_rate_simple(model, a):
    c = model.count
    beta = model.accuracy_simple(a)
    worst_scores = sum(model.queries[model.queries >= model.threshold - a][-c:])
    top_scores = sum(model.queries[:c])
    result = 1 - worst_scores/top_scores, 1 - beta
    print("{} {} {}".format(a, result[0], result[1]))
    return result


def error_rate_precise(model, a):
    c = model.count
    beta = model.accuracy_precise(a)
    worst_scores = sum(model.queries[model.queries >= model.threshold - a][-c:])
    top_scores = sum(model.queries[:c])
    result = 1 - worst_scores/top_scores, beta
    print("{} {} {}".format(a, result[0], result[1]))
    return result


def discrete_pdf(ys):
    return [ys[0]] + [ys[i] - ys[i-1] for i in range(1, len(ys))]


def uniq_xs(pairs_gen):
    pairs_list = []
    dups = defaultdict(list)
    for i, (x, y) in enumerate(pairs_gen):
        dups[x].append(i)
        pairs_list.append((x, y))
    indices = [x for v in dups.values() for x in v[:-1] if len(v) > 1]
    return ((x, y) for i, (x, y) in enumerate(pairs_list) if i not in indices)


if __name__ == '__main__':
    queries = np.loadtxt('data/bms-pos.txt', dtype=int)
    step = 1
    c = 50
    T = threshold(c, queries)
    k = len(queries)

    e = 0.1
    e1, e2 = epsilon(e, c)

    model = Model(T, e1, e2)
    model.queries = queries
    model.response = [True] * c + [False] * (len(queries) - c)
    fig, ax = plt.subplots()
    xs, ys = error_rate_cdf(error_rate, model, step=step)
    ax.bar(xs, ys, color="red", align='edge',
           label=r"$\mathrm{Pr}(\mathrm{SER} \leq x) \geq 1 - \beta$")

    ax.bar(xs, discrete_pdf(ys), width=1/100, color="blue", align='edge',
           label=r"$\mathrm{Pr}(\mathrm{SER} = x) \approx 1 - \beta$")

    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$1 - \beta$")
    ax.legend(loc='upper left')
    plt.title(r"SER distribution on BMS-POS, $k = {}, c = {}, \epsilon = {}$".format(k, c, e))
    plt.show(block=False)

    fig, ax = plt.subplots()
    xs, ys = error_rate_cdf(error_rate_simple, model, step=step)
    ax.bar(xs, ys, color="red", align='edge',
           label=r"$\mathrm{Pr}(\mathrm{SER} \leq x) \geq 1 - \beta$")

    ax.bar(xs, discrete_pdf(ys), width=1/100, color="blue", align='edge',
           label=r"$\mathrm{Pr}(\mathrm{SER} = x) \approx 1 - \beta$")

    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$1 - \beta$")
    ax.legend(loc='upper left')
    plt.title(r"SER distribution on BMS-POS, $k = {}, c = {}, \epsilon = {}$".format(k, c, e))
    plt.show(block=False)

    fig, ax = plt.subplots()
    xs, ys = error_rate_cdf(error_rate_precise, model, step=step)
    ax.bar(xs, ys, color="red", align='edge',
           label=r"$\mathrm{Pr}(\mathrm{SER} \leq x) \geq 1 - \beta$")

    ax.bar(xs, discrete_pdf(ys), width=1/100, color="blue", align='edge',
           label=r"$\mathrm{Pr}(\mathrm{SER} = x) \approx 1 - \beta$")

    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$1 - \beta$")
    ax.legend(loc='upper left')
    plt.title(r"SER distribution on BMS-POS, $k = {}, c = {}, \epsilon = {}$".format(k, c, e))
    plt.show()
