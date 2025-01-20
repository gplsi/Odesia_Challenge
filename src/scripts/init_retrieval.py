import ollama
from langchain_ollama import ChatOllama
import base64
from ollama import Client
from dotenv import load_dotenv
import os
import argparse
from src.data.base import Dataset, DataEncoder
from src.data.tasks import TASK_CONFIG

load_dotenv()  # Loads variables from .env into environment

TASK_CONFIG = {
    "diann_2023": ("tokens", lambda row: " ".join(row["tokens"])),
    "dipromats_2023": ("text", None),
    "exist_2022": ("text", None),
    "exist_2023": ("tweet", None),
    "sqac_squad_2024": ("question", None),
}

def main(args):
    task_key = args.task_key
    partition = args.partition
    language = args.language
    dataset_name, task = task_key[:-3], task_key[-2:]
    text_key, transform = TASK_CONFIG[dataset_name]
    dataset_dir = f"data/{dataset_name}/{partition}_{task}_{language}.json"
    dataset = Dataset.load(dataset_dir, text_key, transform)
    print(f"Dataset len: {len(dataset.items())}")
    # for key, content in dataset.items():
    #     print(f"Key: {key}")
    #     print(f"Content: {content}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Language chain pipeline')
    parser.add_argument('--task_key', type=str, help='Task key', default='diann_2023_t1')
    parser.add_argument('--partition', type=str, help='Partition file', default='train')
    parser.add_argument('--language', type=str, help='Language key', default='es')
    args = parser.parse_args()
    main(args)
