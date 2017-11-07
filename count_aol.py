#!/usr/bin/env python3

import csv
from gzip import open
import string
import re

lenrecords = 0
records = set()
queries = set()
items = set()

separators = '[{}]'.format(string.punctuation + string.whitespace)

for i in range(10):
	filename = 'data/aol/aol-{}.txt.gz'.format(str(i+1).zfill(2))
	with open(filename, 'rt') as file:
		reader = csv.reader(file, delimiter='\t')
		header = next(reader)
		for line in reader:
			anonID, query = line[:2]
			lenrecords += int(int(anonID) not in records)
			records.add(int(anonID))
			queries.add(query)

			items |= set(re.split(separators, query))

			if lenrecords % 1000 == 0:
				output = '\rrecords: {} queries: {} items: {}'.format(
					lenrecords, len(queries), len(items))
				print(output, end='')

print()
print("records:", len(records))
print("queries:", len(queries))
print("items:", len(items))
