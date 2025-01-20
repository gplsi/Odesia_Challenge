# retrieval_service.py
import json
from typing import List, Optional, Dict
from weaviate.collections.classes.filters import Filter
from weaviate.classes.query import MetadataQuery

class RetrievalService:
    def __init__(self, collection, embedding_service):
        self.collection = collection
        self.embedding_service = embedding_service

    def retrieve_examples(
        self,
        query_text: str,
        task_id: str,
        k: int = 5,
        similarity_threshold: float = 0.6
    ) -> List[Dict]:
        query_vec = self.embedding_service.embed_text(query_text)
        
        # Create combined filter for task_id
        task_filter = Filter.by_property("task_id").equal(task_id)
        
        # Execute the query with proper filtering
        response = (
            self.collection.query
            .near_vector(
                near_vector=query_vec,
                limit=k,
                certainty=similarity_threshold,
                filters=task_filter,
                return_metadata=MetadataQuery(distance=True, certainty=True)
            )
            .objects
        )
        
        results = []
        for obj in response:
            results.append({
                "text": obj.properties["text"],
                "content": json.loads(obj.properties["content"]),
                "similarity": obj.metadata.certainty,
                "distance": obj.metadata.distance
            })
        
        return results