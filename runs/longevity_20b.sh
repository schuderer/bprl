#!/bin/bash

echo "Run number $1..."
cd ..
python q_test.py >runs/longevity_20b/longevity_20b_$1.txt

