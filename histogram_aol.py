#!/usr/bin/env python3

from collections import Counter
import csv
from gzip import open as gunzip
import json
import string
import re

separators = '[{}]'.format(string.punctuation + string.whitespace)
data = Counter()

for i in range(10):
	filename = 'data/aol/aol-{}.txt.gz'.format(str(i+1).zfill(2))
	with gunzip(filename, 'rt') as file:
		reader = csv.reader(file, delimiter='\t')
		_header = next(reader)
		data.update(item for line in reader for item in re.split(separators, line[1]))

with open('data/aol.json', 'w') as f:
	json.dump(dict(data), f, indent=0)

