#!/bin/bash

echo "Run number $1..."
cd ..
python q_test.py >runs/longevity_20/longevity_20_$1.txt

