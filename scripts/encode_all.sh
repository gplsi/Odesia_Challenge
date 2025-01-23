#!/bin/bash

# Define arrays for languages, partitions, and task keys
languages=("en" "es")
partitions=("train" "val" "test")
task_keys=("diann_2023_t1" "dipromats_2023_t1" "dipromats_2023_t2" "dipromats_2023_t3" "exist_2022_t1" "exist_2022_t2" "exist_2023_t1" "exist_2023_t2" "exist_2023_t3" "sqac_squad_2024_t1")

# Iterate over each combination of language, partition, and task key
for lang in "${languages[@]}"; do
  for partition in "${partitions[@]}"; do
    for task_key in "${task_keys[@]}"; do
      echo "Processing language: $lang, partition: $partition, task_key: $task_key"
      python -m src.scripts.encode_dataset --language $lang --partition $partition --task_key $task_key --output "$task_key-$lang-$partition.json"
    done
  done
done
