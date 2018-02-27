#!/usr/bin/env python3

from collections import Counter
import json

data = Counter()
with open('data/kosarak.dat') as file:
    data.update(map(int, (item for line in file for item in line.split(' '))))

with open('data/kosarak.json', 'w') as f:
    json.dump(dict(data), f, indent=0)
