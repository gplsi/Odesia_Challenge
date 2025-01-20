# encoder_service.py
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List

class EncoderService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def encode_text(self, text: str) -> List[float]:
        return self.model.embed_query(text)
    
    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)
