#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate chemnlp

for dir in */
do (
    echo "$dir"
    cd "$dir"
    python transform.py
)
done
