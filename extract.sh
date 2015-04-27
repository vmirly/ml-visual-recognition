#!/bin/bash

file=$1

grep '^[^,]' $file | sed 's/,/ /g' | awk '{if($2==1) {printf "%-4d %d\n", NR, $2}}' 
