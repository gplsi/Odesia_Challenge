import json
import ollama
from langchain_ollama import ChatOllama
import base64
from ollama import Client
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
)
from tqdm.auto import tqdm
import os

os.getcwd()
from src.postprocessing.postprocessing import PostProcessingImplementation
from src.evaluation import evaluation
from pathlib import Path


load_dotenv()  # Loads variables from .env into environment


def get_request(ollama_client, instruction, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]
    response = ollama_client.chat(
        model=os.getenv("OLLAMA_MODEL"),
        options={"num_ctx": int(os.getenv("OLLAMA_CONTEXT"))},
        messages=messages,
    )
    print(response["message"]["content"])
    return response["message"]["content"]


def get_client(server: str, username: str, password: str) -> Client:

    print("Initialising ollama client...")

    auth_string = f"{username}:{password}"
    auth_base64 = base64.b64encode(auth_string.encode()).decode()
    headers = {"Authorization": f"Basic {auth_base64}"}
    return Client(host=server, headers=headers)


def get_dataset(task_key, partition, language, text_key, transform=None):
    dataset_name, task = task_key[:-3], task_key[-2:]
    dataset_dir = f"data/{dataset_name}/{partition}_{task}_{language}.json"
    dataset = Dataset.load(dataset_dir, text_key, transform)
    return dataset


def get_overridden_shot_count(task_key, shot_count):
    task_config = TASK_CONFIG[task_key]
    return task_config[K] if K in task_config else shot_count


def get_encoded_data(task_key, partition, language, shot_count):
    answer = partition != "test"
    encoder = DataEncoder(answer)

    task_config = TASK_CONFIG[task_key]
    text_key = task_config[TEXT_KEY]
    class_builder = task_config[CLASS_BUILDER]
    system_prompt = task_config[SYSTEM_PROMPT]
    syntax = task_config[PROMPT_SYNTAX]
    transform = task_config.get(TRANSFORM)

    k = get_overridden_shot_count(task_key, shot_count)
    reRankRetrieval = ReRankRetriever(dataset_id=task_key) if k > 0 else None

    dataset = get_dataset(task_key, partition, language, text_key, transform)

    return encoder.encode(
        dataset, reRankRetrieval, k, class_builder, syntax, system_prompt
    )


def get_encoded_data_from_cache(task_key, partition, language, shot_count, path):
    k = get_overridden_shot_count(task_key, shot_count)
    cache_file = Path(path) / f"{task_key}_{language}_{partition}_{k}.json"
    with open(cache_file, "r") as file:
        return json.load(file)


def main(args):
    task_key = args.task_key
    partition = args.partition
    language = args.language
    shot_count = args.shot_value
    cache_path = args.cache

    encoder_dict = (
        get_encoded_data_from_cache(
            task_key, partition, language, shot_count, cache_path
        )
        if cache_path
        else get_encoded_data(task_key, partition, language, shot_count)
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
        "meta-llama/Llama-3.2-1B-Instruct",
        token=os.getenv("HUGGINGFACE_APIKEY"),
        use_fast=False,
        padding_side="left",
    )

    pipe = pipeline(
        "text-generation",
        model=os.getenv("HUGGINGFACE_MODEL"),
        tokenizer=tokenizer,
        token=os.getenv("HUGGINGFACE_APIKEY"),
        max_length=3000,
        device_map="auto",
    )
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

    model_outputs = []
    generate_kwargs = {
        "do_sample": True,
        "temperature": 1e-3,
        "top_p": 0.9,
    }
    # messages = messages[0:64]

    post_processor = PostProcessingImplementation()
    batch_size = BATCH_SIZE[task_key]
    for i in tqdm(range(0, len(messages), batch_size)):
        batch_data = messages[i : i + batch_size]
        # ids = messages_ids[i:i + batch_size]
        out = pipe(
            batch_data,
            batch_size=batch_size,
            truncation="only_first",
            pad_token_id=pipe.tokenizer.eos_token_id,
            **generate_kwargs,
        )

        # for ids, out in tqdm(
        #     zip(
        #         messages_ids,
        #         pipe(messages, batch_size=BATCH_SIZE[task_key], truncation="only_first", **generate_kwargs),
        #     ),
        #     total=len(messages),
        # ):
        model_outputs.extend(out)

    # print(model_outputs)
    results = [{"id": id, "out": out} for id, out in zip(messages_ids, model_outputs)]

    if task_key == "diann_2023_t1":
        # Generates a json with answers in the correct format to be evaluated
        post_processor.process_ner(results, task_key, language, partition)
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
        post_processor.process_classification(results, task_key, language, partition)
    elif task_key == "sqac_squad_2024_t1":
        # Generates a json with answers in the correct format to be evaluated
        post_processor.process_qa(results, task_key, language, ids)

    # print(results)
    #
    # results = pipe(messages)

    dataset_name = f"{task_key}_{language}"
    predictions_file = f"{task_key}_{language}_{partition}.json"
    gold_file = f"{task_key}_{language}_{partition}_gold.json"
    if task_key == "diann_2023_t1":
        evaluation.evaluate_diann_2023(predictions_file, gold_file, dataset_name)
    if (
        task_key == "dipromats_2023_t1"
        or task_key == "dipromats_2023_t2"
        or task_key == "dipromats_2023_t3"
    ):
        evaluation.evaluate_dipromats_2023(predictions_file, gold_file, dataset_name)
    if task_key == "exist_2022_t1":
        evaluation.evaluate_exist_2022_t1(predictions_file, gold_file, dataset_name)
    if task_key == "exist_2022_t2":
        evaluation.evaluate_exist_2022_t2(predictions_file, gold_file, dataset_name)
    if (
        task_key == "exist_2023_t1"
        or task_key == "exist_2023_t2"
        or task_key == "exist_2023_t3"
    ):
        evaluation.evaluate_exist_2023(predictions_file, gold_file, dataset_name)
    if task_key == "sqac_squad_2024_t1":
        evaluation.evaluate_sqac_squad_2024(predictions_file, gold_file, dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language chain pipeline")
    parser.add_argument(
        "--task_key", type=str, help="Task key", default="sqac_squad_2024_t1"
    )
    parser.add_argument("--partition", type=str, help="Partition file", default="val")
    parser.add_argument("--language", type=str, help="Language key", default="es")
    parser.add_argument("--shot_value", type=str, help="Count of shot", default=0)
    parser.add_argument("--cache", type=str, help="Cache", default="")
    args = parser.parse_args()
    main(args)
