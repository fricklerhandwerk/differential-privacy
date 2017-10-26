#!/usr/bin/env python3

items = set()
with open('data/kosarak.dat') as file:
	for line in file:
		items |= set(map(int, line.split(' ')))

print("items:", len(items))
