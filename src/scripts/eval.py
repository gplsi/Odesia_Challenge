import json
import argparse

from src.evaluation.evaluation import (
    evaluate_diann_2023,
    evaluate_dipromats_2023,
    evaluate_exist_2022_t1,
    evaluate_exist_2022_t2,
    evaluate_exist_2023,
    evaluate_sqac_squad_2024, 
)
from src.postprocessing.postprocessing import PostProcessingImplementation
from src.utils import evaluation_error_handler
from src.data.config import (
    TASK_CONFIG,
    CLASS_BUILDER,
    SYSTEM_PROMPT,
    PROMPT_SYNTAX,
    TEXT_KEY,
    TRANSFORM,
    K,
    BATCH_SIZE,
    EVALUATION,
    USE_BIO
)


def main(args):
    task_key = str(args.task_key)
    partition = args.partition
    language = args.language
    shot_count = args.shot_value
    version = args.version
    results_dir = f"data/model_outputs_{partition}/{task_key}_{language}_{partition}_{shot_count}.json"
    task_config = TASK_CONFIG[task_key][version]

    with open(results_dir, "r") as f:
        results = json.load(f)

    post_processor = PostProcessingImplementation()

    if task_key.startswith("diann_2023"):
        original_data_dir = f"data/diann_2023/{partition}_t1_{language}.json"
        
        with open(original_data_dir, "r") as f:
            original_data = json.load(f)
            
        tokens = [item["tokens"] for item in original_data]
        
        # Generates a json with answers in the correct format to be evaluated
        post_processor.process_ner(tokens, results, task_key, language, partition, shot_count, use_bio_format=task_config[USE_BIO])
    
    elif task_key in (
        "dipromats_2023_t1",
        "dipromats_2023_t2",
        "dipromats_2023_t3",
        "exist_2023_t1",
        "exist_2023_t2",
        "exist_2023_t3",
        "exist_2022_t1",
        "exist_2022_t2",
    ):
        # Generates a json with answers in the correct format to be evaluated
        post_processor.process_classification(
            results, task_key, language, partition, shot_count
        )
        
    elif task_key == "sqac_squad_2024_t1":
        # Generates a json with answers in the correct format to be evaluated
        post_processor.process_qa(results, task_key, language, partition, shot_count)

    else:
        raise ValueError(f"Task key {task_key} not supported")

    dataset_name = f"{task_key}_{language}"
    predictions_file = f"data/results_{partition}/{task_key}_{language}_{partition}_{shot_count}.json"
    
    # regex for eliminating _t1, _t2, _t3 from task key
    #gold_file = f"data_gold/{task_key[:-3]}/{partition}_{task_key[-2:]}_{language}_gold.json"
    gold_file = f"data/data_gold_val/{task_key}_{language}_{partition}_gold.json"
    
    if task_key == "diann_2023_t1":
        evaluate_diann_2023(predictions_file, gold_file, dataset_name, partition)
    
    elif (
        task_key == "dipromats_2023_t1"
        or task_key == "dipromats_2023_t2"
        or task_key == "dipromats_2023_t3"
    ):
        evaluate_dipromats_2023(predictions_file, gold_file, dataset_name, task_config[EVALUATION], partition)
    
    
    elif task_key == "exist_2022_t1":
        evaluate_exist_2022_t1(predictions_file, gold_file, dataset_name, partition)
    
    elif task_key == "exist_2022_t2":
        evaluate_exist_2022_t2(predictions_file, gold_file, dataset_name, partition)
    
    elif (
        task_key == "exist_2023_t1"
        or task_key == "exist_2023_t2"
        or task_key == "exist_2023_t3"
    ):
        evaluate_exist_2023(predictions_file, gold_file, dataset_name, partition)
    
    
    elif task_key == "sqac_squad_2024_t1":
        evaluate_sqac_squad_2024(predictions_file, gold_file, dataset_name, partition)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language chain pipeline")
    parser.add_argument(
        "--task_key", type=str, help="Task key", default="diann_2023_t1"
    )
    parser.add_argument("--partition", type=str, help="Partition file", default="val")
    parser.add_argument("--language", type=str, help="Language key", default="es")
    parser.add_argument("--shot_value", type=str, help="Count of shot", default=0)
    parser.add_argument("--version", type=int, help="Version", default=0)

    args = parser.parse_args()
    main(args)