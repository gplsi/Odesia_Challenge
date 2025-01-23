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
from transformers import pipeline
from src.data.config import (
    TASK_CONFIG,
    CLASS_BUILDER,
    SYSTEM_PROMPT,
    PROMPT_SYNTAX,
    TEXT_KEY,
    TRANSFORM,
    K,
)
from tqdm.auto import tqdm
from src.postprocessing.postprocessing import PostProcessingImplementation
from src.evaluation import evaluation


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


def main(args):
    task_key = args.task_key
    partition = args.partition
    language = args.language
    shot_count = args.shot_value

    reRankRetrieval = ReRankRetriever(dataset_id=task_key)
    answer = partition != "test"
    encoder = DataEncoder(answer)

    task_config = TASK_CONFIG[task_key]
    text_key = task_config[TEXT_KEY]
    class_builder = task_config[CLASS_BUILDER]
    system_prompt = task_config[SYSTEM_PROMPT]
    syntax = task_config[PROMPT_SYNTAX]
    transform = task_config.get(TRANSFORM)

    k = task_config[K] if K in task_config else shot_count

    dataset = get_dataset(task_key, partition, language, text_key, transform)

    encoder_dict = encoder.encode(
        dataset, reRankRetrieval, k, class_builder, syntax, system_prompt
    )

    messages = [
        [
            {"role": "system", "content": encoder_dict[encoder.SYSTEM]},
            {"role": "user", "content": instruction[encoder.USER]},
        ]
        for instruction in encoder_dict[encoder.PROMPTS]
    ]

    pipe = pipeline(
        "text-generation",
        model=os.getenv("HUGGINGFACE_MODEL"),
        token=os.getenv("HUGGINGFACE_APIKEY"),
        max_length=3000,
        device_map="auto",
    )
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

    results = []
    generate_kwargs = {
        "do_sample": True,
        "temperature": 1e-3,
        "top_p": 0.9,
    }

    ids = 0  ########## Sacar este valor del dataset
    for out in tqdm(pipe(messages, batch_size=32, truncation="only_first", **generate_kwargs),total=len(messages)):
        results.append(out)
        if text_key == "diann_2023_t1":
            # Generates a json with answers in the correct format to be evaluated
            PostProcessingImplementation.process_ner(out, text_key, language, ids)
        if (
            text_key == "dipromats_2023_t1"
            or text_key == "dipromats_2023_t2"
            or text_key == "dipromats_2023_t3"
            or text_key == "exist_2023_t1"
            or text_key == "exist_2023_t2"
            or text_key == "exist_2023_t3"
            or text_key == "exist_2022_t1"
            or text_key == "exist_2022_t2"
        ):
            # Generates a json with answers in the correct format to be evaluated
            PostProcessingImplementation.process_classification(out, text_key, language, ids)
        if text_key == "sqac_squad_2024_t1":
            # Generates a json with answers in the correct format to be evaluated
            PostProcessingImplementation.process_qa(out, text_key, language, ids)

    print(results)
    #
    # results = pipe(messages)

    dataset_name=f"{text_key}_{language}"
    predictions_file=f"{text_key}_{language}.json"
    gold_file=f"./../../data/{text_key[:-3]}/gold/{text_key}_{language}_gold.json"
    if text_key == "diann_2023_t1":
        evaluation.evaluate_diann_2023(predictions_file,gold_file,dataset_name)
    if text_key == "dipromats_2023_t1" or text_key == "dipromats_2023_t2" or text_key == "dipromats_2023_t3":
        evaluation.evaluate_dipromats_2023(predictions_file,gold_file,dataset_name)
    if text_key == "exist_2022_t1":
        evaluation.evaluate_exist_2022_t1(predictions_file,gold_file,dataset_name)
    if text_key == "exist_2022_t2":
        evaluation.evaluate_exist_2022_t2(predictions_file,gold_file,dataset_name)
    if text_key == "exist_2023_t1" or text_key == "exist_2023_t2" or text_key == "exist_2023_t3":
        evaluation.evaluate_exist_2023(predictions_file,gold_file,dataset_name)
    if text_key == "sqac_squad_2024_t1":
        evaluation.evaluate_sqac_squad_2024(predictions_file,gold_file,dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language chain pipeline")
    parser.add_argument(
        "--task_key", type=str, help="Task key", default="dipromats_2023_t1"
    )
    parser.add_argument("--partition", type=str, help="Partition file", default="val")
    parser.add_argument("--language", type=str, help="Language key", default="es")
    parser.add_argument("--shot_value", type=str, help="Count of shot", default=0)
    args = parser.parse_args()
    main(args)
