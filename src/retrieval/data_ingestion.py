# data_ingestion.py
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from weaviate.util import generate_uuid5

@dataclass
class Example:
    text: str
    task_id: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]

class DataIngester:
    def __init__(self, collection, batch_size: int = 100):
        self.collection = collection
        self.batch_size = batch_size

    def ingest_examples(self, examples: List[Example]):
        with self.collection.batch.dynamic(batch_size=self.batch_size) as batch:
            for example in examples:
                properties = {
                    "text": example.text,
                    "task_id": example.task_id,
                    "content": json.dumps(example.content),
                    "metadata": example.metadata
                }
                batch.add_object(
                    properties=properties,
                    uuid=generate_uuid5(properties)
                )
