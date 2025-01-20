# main.py
import os
import json

from data_ingestion import DataIngester, Example
from retrieval_service import RetrievalService
from schema import schema
from setup_database import create_weaviate_client, setup_collection

def main():
    # 1. Retrieve Hugging Face API Key from environment variable or another secure source
    # 2. Create Weaviate client pointing to your Docker-based instance
    client = create_weaviate_client(huggingface_api_key=hf_api_key)
    
    # 3. Set up (or reset) TaskExamples collection
    collection = setup_collection(client, schema)
    
    # 4. Create DataIngester and ingest sample data
    ingester = DataIngester(collection)
    sample_examples = [
        Example(
            text="Sample text for multi-task scenario",
            task_id="task_foo",
            content={"any_json": "structure"},
        ),
        Example(
            text="Another example for the same task",
            task_id="task_foo",
            content={"key": "value"},
        )
    ]
    ingester.ingest_examples(sample_examples)

    # 5. Use the RetrievalService to fetch relevant examples
    retriever = RetrievalService(collection)
    results = retriever.retrieve_examples(
        query_text="Similar content to the first sample text",
        task_id="task_123",
        k=2
    )
    
    # 6. Print the retrieved results
    print("Retrieved Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
