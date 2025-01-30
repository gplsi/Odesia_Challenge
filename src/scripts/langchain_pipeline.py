import ollama
from langchain_ollama import ChatOllama
import base64
from ollama import Client
from dotenv import load_dotenv
import os
import argparse
from src.data.base import Dataset, DataEncoder
from src.data import *
#from src.retrieval import ReRankRetriever
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
    EVALUATION
)
from tqdm.auto import tqdm
import os
import json

os.getcwd()
from src.postprocessing.postprocessing import PostProcessingImplementation
#from src.evaluation import evaluation

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


def main(args):
    task_key = args.task_key
    partition = args.partition
    language = args.language
    shot_count = args.shot_value

    #reRankRetrieval = ReRankRetriever(dataset_id=task_key)
    reRankRetrieval = None
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

    messages_ids = [
        instruction[encoder.ID] for instruction in encoder_dict[encoder.PROMPTS]
    ]
    messages = [
        [
            {"role": "system", "content": encoder_dict[encoder.SYSTEM]},
            {"role": "user", "content": instruction[encoder.USER]},
        ]
        for instruction in encoder_dict[encoder.PROMPTS]
    ]
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct",token=os.getenv("HUGGINGFACE_APIKEY"), use_fast=False, padding_side='left')

    pipe = pipeline(
        "text-generation",
        model=os.getenv("HUGGINGFACE_MODEL"),
        tokenizer=tokenizer,
        token=os.getenv("HUGGINGFACE_APIKEY"),
        max_length=5000,
        device_map="auto",
    )
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

    model_outputs = []
    generate_kwargs = {
        "do_sample": True,
        "temperature": 1e-3,
        "top_p": 0.9,
    }
    
    batch_size = BATCH_SIZE[task_key]
    
    for i in tqdm(range(0, len(messages), batch_size)):
        batch_data = messages[i:i + batch_size]
        out = pipe(batch_data, batch_size=batch_size, truncation="only_first", pad_token_id=pipe.tokenizer.eos_token_id, **generate_kwargs)
        model_outputs.extend(out)

    model_outputs = [{"id": id, 'out': out} for id, out in zip(messages_ids, model_outputs)]
    outputs_dir = f"./data/model_outputs_{partition}/{task_key}_{language}_{partition}_{shot_count}.json"
    
    if not os.path.exists(f"./data/model_outputs_{partition}"):
        os.makedirs(f"./data/model_outputs_{partition}")
    
    with open(outputs_dir, "w") as f:
        json.dump(model_outputs, f, indent=4)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language chain pipeline")
    parser.add_argument(
        "--task_key", type=str, help="Task key", default="diann_2023_t1"
    )
    parser.add_argument("--partition", type=str, help="Partition file", default="test")
    parser.add_argument("--language", type=str, help="Language key", default="es")
    parser.add_argument("--shot_value", type=str, help="Count of shot", default=0)
    args = parser.parse_args()
    main(args)
