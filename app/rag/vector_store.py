"""Qdrant vector store initialization and management."""
import uuid
from itertools import zip_longest
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.config import get_settings


class VectorStore:
    """Manages Qdrant vector database operations."""
    
    def __init__(self):
        """Initialize Qdrant client and settings."""
        self.settings = get_settings()
        self.client = QdrantClient(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port
        )
        self.collection_name = self.settings.qdrant_collection
        
    def init_collection(self) -> bool:
        """Initialize the Qdrant collection with proper schema.
        
        Returns:
            True if created, False if already exists
        """
        collections = self.client.get_collections().collections
        if self.collection_name in [col.name for col in collections]:
            return False
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.settings.vector_size,
                distance=Distance.COSINE
            )
        )
        return True
    
    def upsert_vectors(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ):
        """Insert or update vectors in the collection.
        
        Args:
            vectors: List of embedding vectors
            payloads: List of metadata dictionaries
            ids: Optional list of point IDs (auto-generated if None)
        """
        # 1. Validation: Ensure inputs are the same length
        if len(vectors) != len(payloads):
            raise ValueError(f"Length mismatch: {len(vectors)} vectors and {len(payloads)} payloads.")

        # 2. Fix Race Condition: Use UUIDs instead of sequential integers
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        # Validation for manual IDs
        if len(ids) != len(vectors):
            raise ValueError("Length mismatch: IDs list must match vectors list.")
        
        points = [
            PointStruct(id=point_id, vector=vector, payload=payload)
            for point_id, vector, payload in zip(ids, vectors, payloads)
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        category: Optional[str] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in the collection.
        
        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results to return
            category: Optional category filter ('finance', 'geopolitics', 'crypto')
            score_threshold: Minimum similarity score threshold
        
        Returns:
            List of search results with payload and score
        """
        # Build filter if category is specified
        query_filter = None
        if category:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=category)
                    )
                ]
            )
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold
        )
        
        return [{"id": hit.id, "score": hit.score, "payload": hit.payload} for hit in results]
    
    def delete_collection(self):
        """Delete the entire collection (use with caution)."""
        self.client.delete_collection(self.collection_name)


# Global instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the global VectorStore instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
