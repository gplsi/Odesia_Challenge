# main.py
import os
import json

from data_ingestion import DataIngester, Example
from retrieval_service import RetrievalService
from schema import schema
from setup_database import create_weaviate_client, setup_collection
from embedding_service import LocalEmbeddingService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Retrieve Hugging Face API Key from environment variable or another secure source
    # 2. Create Weaviate client pointing to your Docker-based instance
    client = create_weaviate_client()
    
    # 3. Set up (or reset) TaskExamples collection
    collection = setup_collection(client, schema)
    
    # Initialize embedding service
    embedding_service = LocalEmbeddingService()
    
    # 4. Create DataIngester and ingest sample data
    ingester = DataIngester(collection, embedding_service=embedding_service)
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
    
    # Ingest with statistics
    stats = ingester.ingest_examples(sample_examples)
    logger.info(f"Ingestion stats: {stats}")

    # 5. Use the RetrievalService to fetch relevant examples
    retriever = RetrievalService(collection, embedding_service)
    results = retriever.retrieve_examples(
        query_text="Similar content to the first sample text",
        task_id="task_foo",
        k=2
    )
    
    # 6. Print the retrieved results
    logger.info(f"Retrieved {len(results)} results")
    logger.info(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
