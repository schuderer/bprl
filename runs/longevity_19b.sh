#!/bin/bash

echo "Run number $1..."
cd ..
python q_test.py >runs/longevity_19b/longevity_19b_$1.txt

