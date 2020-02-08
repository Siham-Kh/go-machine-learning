#!/bin/bash

folders=$(ls -d -- */)

for x in "30" "50" "100" "1000" "10000" 
do
    for f in $folders
    do
        echo $f
        cd $f
        go run ../parser.go $f $x
        cd ..
    done
done

cd /home/t3st/ML/src/graphs/kernel-01
cat "kernel-30.csv" | shuf > "shuf-kernel-30.csv"
cat "kernel-50.csv" | shuf > "shuf-kernel-50.csv"
cat "kernel-100.csv" | shuf > "shuf-kernel-100.csv"
cat "kernel-1000.csv" | shuf > "shuf-kernel-1000.csv"
cat "kernel-10000.csv" | shuf > "shuf-kernel-10000.csv"