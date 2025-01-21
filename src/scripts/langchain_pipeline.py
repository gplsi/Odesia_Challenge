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


load_dotenv()  # Loads variables from .env into environment

TASK_CONFIG = {
    "diann_2023_t1": {'class_builder': Diann2023T1PromptBuilderBIO},
    "dipromats_2023_t1": {'class_builder': DipromatsT1PromptBuilder("This task (Propaganda Identification) consists on determining whether in a tweet propaganda techniques are used or not. This is a classification task and the labels are 'true' or 'false'."),
                          'system_prompt': 'You are a machine learning model that excels on solving classification problems.', 'syntax_prompt': BasicSyntax(), 'text_key': 'text'},
}


def get_request(ollama_client, instruction, system_prompt):
    messages = [{'role': 'system', 'content': system_prompt},
     {'role': 'user', 'content': instruction}]
    response = ollama_client.chat(model=os.getenv("OLLAMA_MODEL"), options={'num_ctx': int(os.getenv("OLLAMA_CONTEXT"))},
                       messages=messages)
    print(response['message']['content'])
    return response['message']['content']


def get_client(server: str, username: str, password: str) -> Client:

    print('Initialising ollama client...')

    auth_string = f'{username}:{password}'
    auth_base64 = base64.b64encode(auth_string.encode()).decode()
    headers = {
        "Authorization": f'Basic {auth_base64}'
    }
    return Client(host =server, headers =headers)

def main(args):
    task_key = args.task_key
    partition = args.partition
    language = args.language

    dataset_name, task = task_key[:-3], task_key[-2:]
    text_key = TASK_CONFIG[task_key]['text_key']
    dataset_dir = f"data/{dataset_name}/{partition}_{task}_{language}.json"
    dataset = Dataset.load(dataset_dir, text_key)

    reRankRetrieval = ReRankRetriever(dataset_id=task_key)
    answer = partition == 'train'
    encoder = DataEncoder(answer)
    class_builder = TASK_CONFIG[task_key]['class_builder']
    system_prompt = TASK_CONFIG[task_key]['system_prompt']
    syntax = TASK_CONFIG[task_key]['syntax_prompt']
    encoder_dict = encoder.encode(dataset, reRankRetrieval, class_builder, syntax, system_prompt)

    # ollama_client = get_client(os.getenv("OLLAMA_SERVER"), os.getenv("OLLAMA_USERNAME"), os.getenv("OLLAMA_PASSWORD"))
    # responses = [get_request(ollama_client, instruction, encoder_dict['system']) for instruction in
    #              encoder_dict['prompts']]

    messages = [[{'role': 'system', 'content': encoder_dict['system']},
                 {'role': 'user', 'content': instruction}] for instruction in encoder_dict['prompts']]
    pipe = pipeline("text-generation", model=os.getenv("HUGGING_FACE_MODEL"), token=os.getenv("HUGGING_FACE_TOKEN"), max_length=3000, device=0)
    print(messages[0])
    results = pipe(messages[0])
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Language chain pipeline')
    parser.add_argument('--task_key', type=str, help='Task key', default='dipromats_2023_t1')
    parser.add_argument('--partition', type=str, help='Partition file', default='val')
    parser.add_argument('--language', type=str, help='Language key', default='es')
    args = parser.parse_args()
    main(args)
