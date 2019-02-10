#!/bin/bash

echo ""
echo "RUNNING $1 $2 TIMES FOR ANDREAS, PLEASE DON'T CLOSE THIS WINDOW"
echo ""
eval printf "%02d\\\\n" {1..$2} | xargs -n1 -P 25 $1
echo ""
echo "DONE"
echo ""
