"""Embedding service using sentence-transformers."""
from typing import List
from sentence_transformers import SentenceTransformer
from app.config import get_settings


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(get_settings().embedding_model)
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()


_embedder = None

def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder

__all__ = ["Embedder", "get_embedder"]
