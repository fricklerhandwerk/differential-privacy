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

kosarak = sorted(kosarak.values())[::-1]
bms_pos = sorted(bms_pos.values())[::-1]
aol = sorted(aol.values())[::-1]

fig, ax = plt.subplots()
ax.plot(range(len(kosarak)), kosarak, 'r.', label="Kosarak")
ax.plot(range(len(bms_pos)), bms_pos, 'g.', label="BMS-POS")
ax.plot(range(len(aol)), aol, 'b.', label="AOL")

ax.legend(loc='upper center')
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()
