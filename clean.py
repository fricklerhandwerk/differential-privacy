#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
from numpy import log

with open('data/kosarak.json') as f:
	kosarak = json.load(f)

kosarak = sorted(kosarak.values())[::-1]

with open('data/kosarak_clean.json', 'w') as f:
	for line in kosarak:
		f.write('{}\n'.format(line))

del kosarak

with open('data/bms-pos.json') as f:
	bms_pos = json.load(f)

bms_pos = sorted(bms_pos.values())[::-1]

with open('data/bms-pos_clean.json', 'w') as f:
	for line in bms_pos:
		f.write('{}\n'.format(line))

del bms_pos

with open('data/aol.json') as f:
	aol = json.load(f)
	del aol['']

with open('data/stopwords.txt') as f:
	for word in f:
		word = word.strip()
		if word in aol:
			del aol[word]

aol = sorted(aol.values())[::-1]

with open('data/aol_clean.json', 'w') as f:
	for line in aol:
		f.write('{}\n'.format(line))

del aol
