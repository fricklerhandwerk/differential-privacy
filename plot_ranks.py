#!/usr/bin/env python3

from collections import Counter
import json
import matplotlib.pyplot as plt
from numpy import log


with open('data/kosarak.json') as f:
	kosarak = json.load(f)

with open('data/bms-pos.json') as f:
	bms_pos = json.load(f)

with open('data/aol.json') as f:
	aol = json.load(f)

N = 100000
zipf = {n: N//(n+1) for n in range(N)}


def ranks(data):
	rank, score = zip(*((r+1, s) for r, (_, s) in
		                enumerate(Counter(data).most_common())))
	# The zero-th value is needed because the logscale shifts it out
	# of the graph and otherwise we would lose the highest-valued point.
	return (0,) + rank, (0,) + score


fig, ax = plt.subplots()
ax.plot(*ranks(kosarak), 'r', label="Kosarak")
ax.plot(*ranks(bms_pos), 'g', label="BMS-POS")
ax.plot(*ranks(aol), 'b', label="AOL")
ax.plot(*ranks(zipf), 'c', label="Zipf")

ax.legend(loc='upper right')
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlim(1, 2.5*10e1)
plt.ylim(10e1, 10e5)
plt.xlabel("Rank, logscale")
plt.ylabel("Support, logscale")
plt.show()
