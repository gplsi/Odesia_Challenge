script_name=$(basename -- "$0" .sh)
languages=("es")
partition="test"
indexes=(1 0 0 0 0 0 0 0 0 0)
task_keys=("diann_2023_t1" "dipromats_2023_t1" "dipromats_2023_t2" "dipromats_2023_t3" "exist_2022_t1" "exist_2022_t2" "exist_2023_t1" "exist_2023_t2" "exist_2023_t3" "sqac_squad_2024_t1")
shot_value=5
cache_path='./data/test_retrieval/'

# Iterate over each combination of language, partition, and task key
for lang in "${languages[@]}"; do
  for i in "${!task_keys[@]}"; do
    task_key="${task_keys[$i]}"
    version="${indexes[$i]}"
    echo "Processing language: $lang, partition: $partition, task_key: $task_key"
    python -m src.scripts.langchain_pipeline --language $lang --partition $partition  --task_key $task_key --shot_value $shot_value --version $version --tag $script_name --cache $cache_path
    python -m src.scripts.eval --language $lang --partition $partition --task_key $task_key --shot_value $shot_value --version $version --tag $script_name
  done
done
