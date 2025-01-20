# retrieval/embedding_service.py
from typing import List, Union
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
import torch


def get_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = None,
    normalize: bool = True,
) -> Embeddings:
    """
    Creates and returns a HuggingFace embeddings model with specified configuration.

    Args:
        model_name: Name or path of the sentence-transformer model
        device: Device to use (cuda/cpu). If None, automatically detects
        normalize: Whether to normalize embeddings

    Returns:
        A configured HuggingFaceEmbeddings instance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": normalize}

    return HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
