#!/usr/bin/env python3

items = set()
records = 0
with open('data/kosarak.dat') as file:
	for line in file:
		records += 1
		items |= set(map(int, line.split(' ')))

print("records:", records)
print("items:", len(items))
