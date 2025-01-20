# retrieval/rag_service.py
from typing import Any, List, Dict, Tuple
from dataclasses import dataclass
from langchain_community.vectorstores import Weaviate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from weaviate.collections.classes.filters import Filter
import json
import numpy as np
from FlagEmbedding import FlagReranker

@dataclass
class Example:
    text: str
    task_id: str
    content: Dict[str, Any]

@dataclass
class RetrievalStats:
    total_candidates: int
    filtered_candidates: int
    avg_similarity: float
    similarity_std: float
    min_similarity: float
    max_similarity: float

class RAGService:
    def __init__(
        self, 
        vectorstore: Weaviate,
        similarity_threshold: float = 0.6,
        diversity_bias: float = 0.3,
        embedding_function=None
    ):
        self.vectorstore = vectorstore
        self.similarity_threshold = similarity_threshold
        self.diversity_bias = diversity_bias
        self.embedding_function = embedding_function
        
        # Initialize Cohere reranker for better semantic matching
        self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)


    def add_examples(self, examples: List[Example]) -> None:
        texts = [ex.text for ex in examples]
        metadatas = [{
            "content": json.dumps(ex.content),
            "task_id": json.dumps(ex.task_id),
        } for ex in examples]
        
        self.vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas
        )

    def retrieve_examples(
        self, 
        query_text: str,
        k: int = 5,
        fetch_k: int = 20  # Fetch more candidates for better filtering
    ) -> Tuple[List[Dict], RetrievalStats]:
        candidates = self.vectorstore.similarity_search_with_relevance_scores(
            query_text,
            k=fetch_k*2,  # Fetch more for reranking
            score_threshold=self.similarity_threshold
        )
        
        # Prepare for reranking
        texts = [doc.page_content for doc, _ in candidates]
        rerank_inputs = [[query_text, text] for text in texts]
        
        # Rerank using BAAI/bge-reranker-v2-m3
        rerank_scores = self.reranker.compute_score(rerank_inputs)
    
        # Normalize scores to [0,1] range using softmax
        normalized_scores = np.exp(rerank_scores) / np.sum(np.exp(rerank_scores))
        
        # Sort and select top k results using normalized scores
        reranked_results = sorted(
            zip(candidates, normalized_scores), 
            key=lambda x: x[1], 
            reverse=True
        )[:k]
        
        # Calculate similarities for stats
        similarities = [score for _, score in reranked_results]
        
        stats = RetrievalStats(
            total_candidates=len(candidates),
            filtered_candidates=len(reranked_results),
            avg_similarity=float(np.mean(similarities)),
            similarity_std=float(np.std(similarities)),
            min_similarity=float(min(similarities)),
            max_similarity=float(max(similarities))
        )
        
        results = []
        for (doc, _), score in reranked_results:
            result = {
                "text": doc.page_content,
                "content": json.loads(doc.metadata["content"]),
                "task_id": doc.metadata["task_id"],
                "similarity": score,
            }
            results.append(result)
        
        return results, stats

    def batch_retrieve_examples(
        self,
        queries: List[str],
        k: int = 5
    ) -> List[Tuple[List[Dict], RetrievalStats]]:
        """Batch retrieval for multiple queries"""
        return [
            self.retrieve_examples(query, k)
            for query in queries
        ]
