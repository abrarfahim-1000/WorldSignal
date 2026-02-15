"""Qdrant vector store initialization and management."""
import uuid
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
    def __init__(self):
        self.settings = get_settings()
        self.client = QdrantClient(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port
        )
        self.collection_name = self.settings.qdrant_collection
        
    def init_collection(self, force_recreate: bool = False) -> bool:
        collections = self.client.get_collections().collections
        collection_exists = self.collection_name in [col.name for col in collections]
        
        if collection_exists:
            if force_recreate:
                self.delete_collection()
            else:
                # Check if dimensions match
                collection_info = self.client.get_collection(self.collection_name)
                existing_dim = collection_info.config.params.vectors.size
                if existing_dim != self.settings.vector_size:
                    print(f"⚠ Warning: Collection dimension mismatch. Expected {self.settings.vector_size}, got {existing_dim}")
                    print(f"  Recreating collection with correct dimensions...")
                    self.delete_collection()
                    collection_exists = False
                else:
                    return False
        
        if not collection_exists or force_recreate:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.settings.vector_size,
                    distance=Distance.COSINE
                )
            )
            return True
        return False
    
    def upsert_vectors(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ):
        if len(vectors) != len(payloads):
            raise ValueError(f"Length mismatch: {len(vectors)} vectors and {len(payloads)} payloads.")

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        elif len(ids) != len(vectors):
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
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold
        ).points
        
        return [{"id": hit.id, "score": hit.score, "payload": hit.payload} for hit in results]
    
    def delete_collection(self):
        self.client.delete_collection(self.collection_name)


_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

__all__ = ["VectorStore", "get_vector_store"]
