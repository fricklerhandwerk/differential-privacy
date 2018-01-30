#!/usr/bin/env python3

from collections import Counter
import csv
from gzip import open as gunzip
import json
import string
import re

separators = '[{}]'.format(string.punctuation + string.whitespace)
queries = set()

for i in range(10):
	filename = 'data/aol/aol-{}.txt.gz'.format(str(i+1).zfill(2))
	with gunzip(filename, 'rt') as file:
		reader = csv.reader(file, delimiter='\t')
		_header = next(reader)
		queries |= set(line[1] for line in reader)

data = Counter(item for query in queries for item in re.split(separators, query))

data = dict(data)

# filter out stopwords
with open('data/stopwords.txt') as f:
	for word in f:
		word = word.strip()
		if word in data:
			del data[word]

with open('data/aol.json', 'w') as f:
	json.dump(data, f, indent=0)
