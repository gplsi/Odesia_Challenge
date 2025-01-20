# data_ingestion.py
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from weaviate.util import generate_uuid5
from embedding_service import HuggingFaceEmbeddings
from weaviate.collections.classes.filters import Filter

@dataclass
class Example:
    text: str
    task_id: str
    content: Dict[str, Any]

class DataIngester:
    def __init__(self, collection, embedding_service, batch_size: int = 100):
        self.collection = collection
        self.batch_size = batch_size
        self.embedding_service = embedding_service

    def ingest_examples(self, examples: List[Example]) -> Dict[str, int]:
        """
        Ingest examples with duplicate checking.
        Returns statistics about the ingestion process.
        """
        stats = {"processed": 0, "duplicates": 0, "added": 0}
        
        with self.collection.batch.dynamic() as batch:
            for example in examples:
                stats["processed"] += 1
                
                vec = self.embedding_service.embed_text(example.text)
                properties = {
                    "text": example.text,
                    "task_id": example.task_id,
                    "content": json.dumps(example.content),
                }
                
                batch.add_object(
                    properties=properties,
                    uuid=generate_uuid5(properties),
                    vector=vec
                )
                stats["added"] += 1
        
        return stats
