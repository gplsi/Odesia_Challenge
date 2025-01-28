#!/bin/bash

# Define arrays for languages, partitions, and task keys
languages=("en")  # ["en", "es"]
partition="val"   # ["val", "test"]
# task_keys=("diann_2023_t1" "dipromats_2023_t1" "dipromats_2023_t2" "dipromats_2023_t3" "exist_2022_t1" "exist_2022_t2" "exist_2023_t1" "exist_2023_t2" "exist_2023_t3" "sqac_squad_2024_t1")
task_keys=("dipromats_2023_t1")
shot_value=0

# Iterate over each combination of language, partition, and task key
for lang in "${languages[@]}"; do
  for task_key in "${task_keys[@]}"; do
    echo "Processing language: $lang, partition: $partition, task_key: $task_key"
    python -m src.scripts.langchain_pipeline --language $lang --partition $partition --task_key $task_key --shot_value $shot_value
  done
done
