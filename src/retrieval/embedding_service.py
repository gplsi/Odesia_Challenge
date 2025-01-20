# encoder_service.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
import torch

class LocalEmbeddingService:
    def __init__(self, model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name_or_path
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name_or_path,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text."""
        return self.embedding_model.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        return self.embedding_model.embed_documents(texts)