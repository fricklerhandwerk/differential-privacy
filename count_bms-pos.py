#!/usr/bin/env python3

import csv

records = set()
items = set()
with open('data/bms-pos.dat') as file:
    reader = csv.reader(file, delimiter='\t')
    for line in reader:
        record, item = map(int, line)
        records.add(record)
        items.add(item)

print("records:", len(records))
print("items:", len(items))
