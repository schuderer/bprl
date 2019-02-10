#!/bin/bash

echo "Run number $1..."
cd ..
python q_test.py >runs/longevity_19/longevity_19_$1.txt

