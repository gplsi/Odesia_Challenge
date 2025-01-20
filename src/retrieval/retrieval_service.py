# retrieval_service.py
import json
from typing import List, Optional

class RetrievalService:
    def __init__(self, collection):
        self.collection = collection

    def retrieve_examples(
        self, 
        query_text: str, 
        task_id: str, 
        k: int = 5
    ) -> List[dict]:
        response = self.collection.query.near_text(
            query=query_text,
            limit=k,
            filters={
                "path": ["task_id"],
                "operator": "Equal",
                "valueText": task_id
            }
        )
        
        results = []
        for obj in response.objects:
            results.append({
                "text": obj.properties["text"],
                "content": json.loads(obj.properties["content"]),
                "metadata": obj.properties["metadata"]
            })
        
        return results
