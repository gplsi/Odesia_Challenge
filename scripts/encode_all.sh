#!/bin/bash

# Define arrays for languages, partitions, and task keys
languages=("es")
partitions=("test")
task_keys=("diann_2023_t1")

# Iterate over each combination of language, partition, and task key
for lang in "${languages[@]}"; do
  for partition in "${partitions[@]}"; do
    for task_key in "${task_keys[@]}"; do
      echo "Processing language: $lang, partition: $partition, task_key: $task_key"
      python -m src.scripts.encode_dataset --language $lang --partition $partition --task_key $task_key --shot_value 5 --version 3
    done
  done
done
