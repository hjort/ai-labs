#!/bin/bash

for a in *.csv; do echo "[$a]:"; diff $a ../iris-solution.csv ; echo; done | less
