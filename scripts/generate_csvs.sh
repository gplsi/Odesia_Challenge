#!/bin/bash
folder_in="/workspace/data/results"
folder_out="/workspace/data/"
# Define arrays for languages, partitions, and task keys
python3 src/scripts/generate_csvs.py --folder_in "$folder_in" --folder_out "$folder_out"