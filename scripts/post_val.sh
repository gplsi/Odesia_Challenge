#!/bin/bash

# Define arrays for languages, partitions, and task keys
languages=("es")  # ["en", "es"]
partition=("val")   # ["val", "test"]
#indexes=(1 0 0 0 0 0 0 0 0 0)
indexes=(2)
#task_keys=("diann_2023_t1" "dipromats_2023_t1" "dipromats_2023_t2" "dipromats_2023_t3" "exist_2022_t1" "exist_2022_t2" "exist_2023_t1" "exist_2023_t2" "exist_2023_t3" "sqac_squad_2024_t1")
#task_keys=("sqac_squad_2024_t1")
task_keys=("diann_2023_t1")
#task_keys=("dipromats_2023_t1" "dipromats_2023_t2" "dipromats_2023_t3")
#task_keys=("exist_2022_t1")
shot_value=0
script_name="02_finetuned_llama3b_instruct_0shot_nonbio"

# Iterate over each combination of language, partition, and task key
for lang in "${languages[@]}"; do
  for i in "${!task_keys[@]}"; do
    task_key="${task_keys[$i]}"
    version="${indexes[$i]}"
    echo "Processing language: $lang, partition: $partition, task_key: $task_key"
    #python -m src.scripts.eval --language $lang --partition $partition --task_key $task_key --shot_value $shot_value --version $version
    python -m src.scripts.eval --language $lang --partition $partition --task_key $task_key --shot_value $shot_value --version $version --tag $script_name
  done
done
