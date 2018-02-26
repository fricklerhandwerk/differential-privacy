#!/usr/bin/env python3

from collections import Counter
import json
import matplotlib.pyplot as plt
from math import log
from math import isclose
import numpy as np
from numpy import product
from scipy.integrate import quad
from scipy.stats import rv_discrete

from accuracy import probability_precise
from accuracy import accuracy_optimized
from algorithms import scale
from algorithms import epsilon
from algorithms import factor
from algorithms import Laplace
from naive import sparse


datasets = {
    'bms-pos': "BMS-POS",
    'kosarak': "Kosarak",
    'aol': "AOL",
    'zipf': "Zipf",
}

e = 0.1  # this is what [@svt] used
cs = list(range(25, 301, 25))
b_min = 0.01  # only compute probabilities down to this value
N = 100


def threshold(c, queries):
    return (queries[c-1] + queries[c])/2


def ratios(c, monotonic=True):
    m = factor(monotonic)
    return {
        '1': 1,
        '3': 3,
        'c': m*c,
        'c23': (m*c)**(2/3),
    }


def read_data(data):
    with open('data/{}.json'.format(data)) as f:
        items = json.load(f, object_hook=convert)
        counts = Counter(items.values())
    array = np.loadtxt('data/{}.txt'.format(data), dtype=int)
    return counts, array


def above(queries, T, a):
    return {k: v for k, v in queries.items() if k >= T + a}


def below(queries, T, a):
    return {k: v for k, v in queries.items() if k <= T - a}


def write_alphas(data, start=None, end=None):
    # compute probabilities only for unique tuples with numbers of queries
    # above and below the T+/-alpha range
    query_counts, query_array = read_data(data)
    k = query_array[0]

    print("loaded {}, max. value: {}".format(data, k))

    for c in cs[start:end]:
        T = threshold(c, query_array)
        print('T:', T, end=' ')

        above_below = {}
        for a in range(k):
            queries_below = below(query_counts, T, a)
            queries_above = above(query_counts, T, a)
            key = (len(queries_below.keys()), len(queries_above.keys()))
            above_below[key] = {
                'alpha': a,
                'below': queries_below,
                'above': queries_above,
            }

        print('c:', c, len(above_below), end='\r')

        result = {}
        for _, v in above_below.items():
            result[v['alpha']] = {
                'below': v['below'],
                'above': v['above'],
            }

        with open('experiments/{}-alphas {}.txt'.format(data, c), 'w') as f:
            json.dump(result, f, separators=(',', ':'))


def read_alphas(data, c):
    with open('experiments/{}-alphas {}.txt'.format(data, c)) as f:
        return json.load(f, object_hook=convert)


def convert(d):
    result = {}
    for k, v in d.items():
        if k.isdigit():
            result[int(k)] = v
        else:
            result[k] = v
    return result


def basic(a, k, s1, s2, *args):
    return probability_precise(a, k, s1, s2)


def precise(a, k, s1, s2, queries, alphas, T):
    below = alphas[a]['below']
    above = alphas[a]['above']

    # IMPORTANT: we *must* compute the probability of a correct response here,
    # although eventually we want to have the error probability.
    # the reason is that queries are usually so far away from the threshold that
    # getting a wrong response is incredibly improbable, which in effect produces
    # zeroes instead of *very very* small numbers. and by small I mean so small
    # that even using gmpy2 with precision 10000 doesn't capture them properly.
    rs = [False] * len(below) + [True] * len(above)
    qs = {**below, **above}

    def pred(x):
        return product([query_above(s2, q, r, x)**n for (r, (q, n)) in zip(rs, qs.items())])

    def state(x):
        return Laplace(s1, T).pdf(x) * pred(x)

    # integrate over some sufficiently large quantile of the threshold distribution,
    # here we take everything except a 2*error part.
    # bounds of [0, max(queries)] as I had them before will give catastrophically wrong results.
    error = 1/1e12
    T_bound = s1 * log(1/error)
    return 1 - quad(state, T-T_bound, T+T_bound, points=[T])[0]


def query_above(scale, loc, is_above, threshold):
    """Pr(query(scale, loc) => is_above | threshold_value )"""
    pr_above = 1 - Laplace(scale, loc).cdf(threshold)
    if is_above:
        return pr_above
    else:
        return 1 - pr_above


def write_probability(data, func, start=None, end=None):
    query_counts, query_array = read_data(data)
    k = query_array[0]
    for c in cs[start:end]:
        T = threshold(c, query_array)
        alphas = read_alphas(data, c)
        total = len(alphas)

        for s, r in ratios(c).items():
            print("c: {}, r: {}".format(c, s))
            s1, s2 = scale(*epsilon(e, r), c)
            last = 1
            for i, a in enumerate(alphas.keys()):
                p = func(a, k, s1, s2, query_array, alphas, T)
                # catch problems with integration
                if p > last or (p == last and p < 1) or p <= 0:
                    break
                else:
                    last = p
                print(total - i, a, p, end='\r')
                with open('experiments/{}-{} {} {}.txt'.format(data, func.__name__, c, s), 'a') as f:
                    print(a, p, file=f)
            print()


def write_samples(data):
    """recreate the experiments from [@svt]"""
    database = np.loadtxt('data/{}.txt'.format(data), dtype=int)

    for n in range(N):
        print(n)
        queries = np.random.permutation(range(len(database)))
        for c in cs:
            print(c, end='\r')
            T = threshold(c, database)
            for s, r in ratios(c).items():
                ser = []
                response = sparse(database, queries, T, e, r, c)
                ser.append(score_error_rate(database, queries, response, c))
                with open('experiments/{}-samples {} {}.txt'.format(data, c, s), 'a') as f:
                    for x in ser:
                        print(x, file=f)
        print()


def score_error_rate(database, queries, response, c):
    best_avg = sum(database[:c]) / c
    sampled_score = sum(database[q] for q, x in zip(queries, response) if x)
    sampled_length = len([x for x in response if x])
    sampled_avg =  sampled_score / sampled_length
    return 1 - sampled_avg / best_avg


def plot_samples(data):
    fig, ax = plt.subplots(figsize=(5,3))
    colors = ['black', 'magenta', 'blue', 'red']
    for s, color in zip(ratios(1).keys(), colors):
        ys = []
        std = []
        for c in cs:
            samples = np.loadtxt('experiments/{}-samples {} {}.txt'.format(data, c, s))
            ys.append(np.mean(samples))
            std.append((0, np.std(samples)))
        ax.errorbar(cs, ys, yerr=list(zip(*std)), color=color, capsize=5, fmt='-o', barsabove=True)

    ax.set_aspect(150) # this value makes no sense to me
    plt.xlim(min(cs),max(cs))
    plt.ylim(0,1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xticks(cs)
    plt.xlabel("c")
    plt.ylabel("SER")
    plt.title("{}, SER samples".format(datasets[data]))
    plt.show()


def plot_accuracy(data, func):
    fig, ax = plt.subplots(figsize=(5,3))
    colors = ['black', 'magenta', 'blue', 'red']
    counts, array = read_data(data)
    for s, color in zip(ratios(1).keys(), colors):
        ys = []
        std = []
        for c in cs:
            results = np.genfromtxt('experiments/{}-{} {} {}.txt'.format(data, func.__name__, c, s), dtype=None)
            ser, prob = to_pdf(*to_cdf(results, c, array))
            # need to handle the case where the estimation is too bad
            # and does not lead to a proper distribution
            if isclose(sum(prob), 1, rel_tol=1e-05):
                rv = rv_discrete(values=(ser, prob))
                ys.append(rv.mean())
                std.append((0, rv.std()))
            else:
                ys.append(1)
                std.append((0, 0))

        ax.errorbar(cs, ys, yerr=list(zip(*std)), color=color, capsize=5, fmt='-o', barsabove=True)

    ax.set_aspect(150) # this value makes no sense to me
    plt.xlim(min(cs),max(cs))
    plt.ylim(0,1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xticks(cs)
    plt.xlabel("c")
    plt.ylabel("SER")
    plt.title("{}, SER estimation".format(datasets[data]))
    plt.show()


def plot_accuracy_alpha(data, func):
    fig, ax = plt.subplots(figsize=(5,3))
    xs = []
    ys = []
    pr = []
    for c in cs:
        counts, array = read_data(data)
        results = np.genfromtxt('experiments/{}-{} {} c23.txt'.format(data, func.__name__, c), dtype=None)
        results = np.atleast_1d(results) # https://stackoverflow.com/a/24247766
        ser, prob = to_pdf(*to_cdf(results, c, array))
        prob = discrete_pdf(prob)
        colors = np.zeros((len(ser), 4))
        colors[:,0] = 1 # red
        colors[:,3] = prob # alpha
        ax.scatter([c] * len(ser), ser, color=colors, marker='s', edgecolors='none')

    ax.set_aspect(150) # this value makes no sense to me
    plt.xlim(min(cs),max(cs))
    plt.ylim(0,1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xticks([1] + cs)
    plt.xlabel("c")
    plt.ylabel("SER")
    plt.title("{}, SER estimation".format(datasets[data]))
    plt.show()


def to_cdf(results, c, database):
    ser = []
    prob = []
    for a, b in results:
        ser.append(score_error_rate_alpha(database, c, a))
        prob.append(1 - b)
    return ser, prob


def to_pdf(ser, prob):
    return ser, discrete_pdf(prob)


def score_error_rate_alpha(database, c, a):
    T = threshold(c, database)
    # the length is always c, so we don't need to divide
    best_case = sum(database[:c])
    worst_case = sum(database[database >= T - a][-c:])
    return 1 - worst_case / best_case


def discrete_pdf(ys):
    return [ys[0]] + [ys[i] - ys[i-1] for i in range(1, len(ys))]
