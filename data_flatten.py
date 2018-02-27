#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
from numpy import log

with open('data/kosarak.json') as f:
	kosarak = json.load(f)

kosarak = sorted(kosarak.values())[::-1]

with open('data/kosarak.txt', 'w') as f:
	for line in kosarak:
		f.write('{}\n'.format(line))

del kosarak

with open('data/bms-pos.json') as f:
	bms_pos = json.load(f)

bms_pos = sorted(bms_pos.values())[::-1]

with open('data/bms-pos.txt', 'w') as f:
	for line in bms_pos:
		f.write('{}\n'.format(line))

del bms_pos

with open('data/aol.json') as f:
	aol = json.load(f)

aol = sorted(aol.values())[::-1]

with open('data/aol.txt', 'w') as f:
	for line in aol:
		f.write('{}\n'.format(line))

del aol
