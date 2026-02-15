"""RAG retriever for querying the vector store."""
from typing import List, Dict, Any, Optional
from app.rag.vector_store import get_vector_store
from app.embeddings.base import get_embedder


class Retriever:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.embedder = get_embedder()
    
    def retrieve(
        self,
        query: str,
        limit: int = 5,
        category: Optional[str] = None,
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        query_vector = self.embedder.embed([query])[0]
        
        results = self.vector_store.search(
            query_vector=query_vector,
            limit=limit,
            category=category,
            score_threshold=score_threshold
        )
        
        return results


_retriever = None

def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever

__all__ = ["Retriever", "get_retriever"]
