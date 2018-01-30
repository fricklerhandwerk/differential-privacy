#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
from numpy import log

with open('data/kosarak.json') as f:
	kosarak = json.load(f)

with open('data/bms-pos.json') as f:
	bms_pos = json.load(f)

with open('data/aol.json') as f:
	aol = json.load(f)

# The zero-th value is needed because the logscale shifts it out
# of the graph and otherwise we would lose the highest-valued point.
kosarak = [0] + sorted(kosarak.values())[::-1]
bms_pos = [0] + sorted(bms_pos.values())[::-1]
aol = [0] + sorted(aol.values())[::-1]
N = 100000
zipf = [0] + [N//(n+1) for n in range(N)]

fig, ax = plt.subplots()
ax.plot(range(len(kosarak)), kosarak, 'r', label="Kosarak")
ax.plot(range(len(bms_pos)), bms_pos, 'g', label="BMS-POS")
ax.plot(range(len(aol)), aol, 'b', label="AOL")
ax.plot(range(len(zipf)), zipf, 'c', label="Zipf")

ax.legend(loc='upper right')
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlim(1, 2.5*10e1)
plt.ylim(10e1, 10e5)
plt.xlabel("Rank, logscale")
plt.ylabel("Support, logscale")
plt.show()
