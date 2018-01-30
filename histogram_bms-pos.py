#!/usr/bin/env python3

from collections import Counter
import csv
import json

data = Counter()
with open('data/BMS-POS.dat') as file:
    reader = csv.reader(file, delimiter='\t')
    data.update((int(item) for _record, item in reader))

with open('data/bms-pos.json', 'w') as f:
    json.dump(dict(data), f, indent=0)
