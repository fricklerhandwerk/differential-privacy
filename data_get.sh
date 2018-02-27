#!/bin/sh

mkdir -p data temp

wget http://www.kdd.org/cupfiles/KDDCup2000.zip -P temp
unzip temp/KDDCup2000.zip assoc/BMS-POS.dat.gz -d temp
gunzip temp/assoc/BMS-POS.dat.gz -c > data/bms-pos.dat

wget http://fimi.ua.ac.be/data/kosarak.dat -P data

wget https://archive.org/download/AOL_search_data_leak_2006/AOL_search_data_leak_2006.zip -P temp
unzip temp/AOL_search_data_leak_2006.zip AOL-user-ct-collection/* -d temp
mkdir -p data/aol
mv temp/AOL-user-ct-collection/*.gz data/aol/

rm -rf temp

