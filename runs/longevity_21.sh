#!/bin/bash

echo "Run number $1..."
cd ..
python q_test.py >runs/longevity_21/longevity_21_$1.txt

