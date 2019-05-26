#!/bin/bash

file *.csv
for a in *.csv; do sed -i 's/\r//g' $a; done
sed -i 's/"//g' *flavia*

file *.csv
wc -l *.csv
ls -la *.csv

