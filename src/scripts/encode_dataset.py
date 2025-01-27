import json
import argparse
from src.data.base import Dataset, DataEncoder
from src.data.config import (
    TASK_CONFIG,
    CLASS_BUILDER,
    SYSTEM_PROMPT,
    PROMPT_SYNTAX,
    TEXT_KEY,
    TRANSFORM,
    K,
)
from src.retrieval import ReRankRetriever


def get_dataset(task_key, partition, language, text_key, transform=None):
    dataset_name, task = task_key[:-3], task_key[-2:]
    dataset_dir = f"data/{dataset_name}/{partition}_{task}_{language}.json"
    dataset = Dataset.load(dataset_dir, text_key, transform)
    return dataset


def main(args):
    task_key = args.task_key
    partition = args.partition
    language = args.language
    output_path = args.output
    shot_count = args.shot_value

    task_config = TASK_CONFIG[task_key]
    text_key = task_config[TEXT_KEY]
    class_builder = task_config[CLASS_BUILDER]
    system_prompt = task_config[SYSTEM_PROMPT]
    prompt_syntax = task_config[PROMPT_SYNTAX]
    transform = task_config.get(TRANSFORM)
    k = task_config[K] if K in task_config else shot_count

    dataset = get_dataset(task_key, partition, language, text_key, transform)
    reRankRetrieval = ReRankRetriever(dataset_id=task_key) if k > 0 else None
    encoder = DataEncoder(True)

    encoded = encoder.encode(
        dataset,
        reRankRetrieval,
        k,
        class_builder,
        prompt_syntax,
        system_prompt,
    )

    if output_path is None:
        print(json.dumps(encoded, indent=2))
    else:
        with open(output_path, "w") as fd:
            json.dump(encoded, fd, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language chain pipeline")
    parser.add_argument(
        "--task_key", type=str, help="Task key", default="dipromats_2023_t1"
    )
    parser.add_argument("--partition", type=str, help="Partition file", default="val")
    parser.add_argument("--language", type=str, help="Language key", default="es")
    parser.add_argument("--shot_value", type=str, help="Count of shot", default=0)
    parser.add_argument("--output", type=str, help="Output file", default="dipromats_2023_t1.json")
    args = parser.parse_args()
    main(args)
