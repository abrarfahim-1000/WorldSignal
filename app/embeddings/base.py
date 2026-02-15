"""Embedding service using Gemini or OpenAI API."""
from typing import List
from app.config import get_settings


class Embedder:
    def __init__(self):
        settings = get_settings()
        if settings.gemini_api_key:
            from google import genai
            self.client = genai.Client(api_key=settings.gemini_api_key)
            self.backend = "gemini"
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
            self.backend = "openai"
        self.model = settings.embedding_model
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.backend == "gemini":
            # result = self.client.models.embed_content(model=self.model, contents=texts)
            return [
                self.client.models.embed_content(
                    model=self.model, 
                    contents=text
                ).embeddings[0].values 
                for text in texts
                ]
        else:
            response = self.client.embeddings.create(input=texts, model=self.model)
            return [item.embedding for item in response.data]


_embedder = None

def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder

__all__ = ["Embedder", "get_embedder"]
