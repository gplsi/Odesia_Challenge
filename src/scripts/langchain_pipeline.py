import json
import base64
from dotenv import load_dotenv
import os
import argparse
from src.data.base import Dataset, DataEncoder
from src.data import *

from src.retrieval import ReRankRetriever
from transformers import pipeline, AutoTokenizer
from src.data.config import (
    TASK_CONFIG,
    CLASS_BUILDER,
    SYSTEM_PROMPT,
    PROMPT_SYNTAX,
    TEXT_KEY,
    TRANSFORM,
    K,
    BATCH_SIZE,
    RELATIVE_BATCH_SIZE,
    EVALUATION,
)
from tqdm.auto import tqdm
import os
import json

os.getcwd()
from src.postprocessing.postprocessing import PostProcessingImplementation
from pathlib import Path

from src.utils import evaluation_error_handler

from src.evaluation.evaluation import (
    evaluate_diann_2023,
    evaluate_dipromats_2023,
    evaluate_exist_2022_t1,
    evaluate_exist_2022_t2,
    evaluate_exist_2023,
    evaluate_sqac_squad_2024,
)

load_dotenv()  # Loads variables from .env into environment


def get_dataset(task_key, partition, language, text_key, transform=None):
    dataset_name, task = task_key[:-3], task_key[-2:]
    dataset_dir = f"data/{dataset_name}/{partition}_{task}_{language}.json"
    dataset = Dataset.load(dataset_dir, text_key, transform)
    return dataset


def get_overridden_shot_count(task_key, shot_count, version):
    task_config = TASK_CONFIG[task_key][version]
    return task_config[K] if K in task_config else shot_count


def get_encoded_data(task_key, partition, language, shot_count, version):
    answer = partition != "test"
    encoder = DataEncoder(answer)

    task_config = TASK_CONFIG[task_key][version]
    text_key = task_config[TEXT_KEY]
    class_builder = task_config[CLASS_BUILDER]
    system_prompt = task_config[SYSTEM_PROMPT]
    syntax = task_config[PROMPT_SYNTAX]
    transform = task_config.get(TRANSFORM)

    k = get_overridden_shot_count(task_key, shot_count, version)
    reRankRetrieval = ReRankRetriever(dataset_id=task_key) if k > 0 else None

    dataset = get_dataset(task_key, partition, language, text_key, transform)

    return encoder.encode(
        dataset, reRankRetrieval, k, class_builder, syntax, system_prompt
    )


def get_encoded_data_from_cache(
    task_key, partition, language, shot_count, path, version
):
    k = get_overridden_shot_count(task_key, shot_count, version)
    cache_file = Path(path) / f"{task_key}-{language}-{partition}-{k}.json"
    print(f"Loading data from cache: {cache_file}")
    with open(cache_file, "r") as file:
        return json.load(file)


def main(args):
    task_key = args.task_key
    partition = args.partition
    language = args.language
    shot_count = args.shot_value
    cache_path = args.cache
    version = args.version
    use_global_batch_size = args.use_global_batch_size
    tag = args.tag

    encoder_dict = (
        get_encoded_data_from_cache(
            task_key, partition, language, shot_count, cache_path, version
        )
        if cache_path
        else get_encoded_data(task_key, partition, language, shot_count, version)
    )

    messages_ids = [
        instruction[DataEncoder.ID] for instruction in encoder_dict[DataEncoder.PROMPTS]
    ]
    messages = [
        [
            {"role": "system", "content": encoder_dict[DataEncoder.SYSTEM]},
            {"role": "user", "content": instruction[DataEncoder.USER]},
        ]
        for instruction in encoder_dict[DataEncoder.PROMPTS]
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        token=os.getenv("HUGGINGFACE_APIKEY"),
        use_fast=False,
        padding_side="left",
    )

    pipe = pipeline(
        "text-generation",
        model=os.getenv("HUGGINGFACE_MODEL"),
        tokenizer=tokenizer,
        token=os.getenv("HUGGINGFACE_APIKEY"),
        max_length=10000,
        device_map="auto",
    )
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

    model_outputs = []
    generate_kwargs = {
        "do_sample": True,
        "temperature": 1e-3,
        "top_p": 0.9,
    }

    batch_size = (
        BATCH_SIZE[task_key]
        if use_global_batch_size
        else RELATIVE_BATCH_SIZE[task_key][version][shot_count]
    )

    for i in tqdm(range(0, len(messages), batch_size)):
        batch_data = messages[i : i + batch_size]
        out = pipe(
            batch_data,
            batch_size=batch_size,
            truncation="only_first",
            pad_token_id=pipe.tokenizer.eos_token_id,
            **generate_kwargs,
        )
        model_outputs.extend(out)

    model_outputs = [
        {"id": id, "out": out} for id, out in zip(messages_ids, model_outputs)
    ]
    outputs_dir = f"./data/{tag}/model_outputs_{partition}/{task_key}_{language}_{partition}_{shot_count}.json"

    if not os.path.exists(f"./data/{tag}/model_outputs_{partition}"):
        os.makedirs(f"./data/{tag}/model_outputs_{partition}", exist_ok=True)

    with open(outputs_dir, "w") as f:
        json.dump(model_outputs, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language chain pipeline")
    parser.add_argument(
        "--task_key", type=str, help="Task key", default="diann_2023_t1"
    )
    parser.add_argument("--partition", type=str, help="Partition file", default="test")
    parser.add_argument("--language", type=str, help="Language key", default="es")
    parser.add_argument("--shot_value", type=int, help="Count of shot", default=0)
    parser.add_argument("--cache", type=str, help="Cache", default="")
    parser.add_argument("--version", type=int, help="Version", default=0)
    parser.add_argument("--use-global-batch-size", action="store_true")
    parser.add_argument("--tag", type=str, help="Tag", default=".")
    args = parser.parse_args()
    main(args)
