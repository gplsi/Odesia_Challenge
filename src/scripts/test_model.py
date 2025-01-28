import json
import os
import pandas as pd
from transformers import pipeline
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
model = AutoModelForCausalLM.from_pretrained(os.getenv("HUGGINGFACE_MODEL"), token=os.getenv("HUGGING_FACE_TOKEN"))
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct",token=os.getenv("HUGGING_FACE_TOKEN"), use_fast=False)


pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=3000, device=0, use_fast=False)
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
 ]
results = pipe(messages)
print(results)