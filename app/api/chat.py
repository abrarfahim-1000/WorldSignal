"""FastAPI application with chat endpoint, RAG and streaming support."""
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.rag.retriever import get_retriever
from app.llm.base import get_llm_service
from app.db.models import init_db
from app.rag.vector_store import get_vector_store


app = FastAPI(title="WorldSignal RAG API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    init_db()
    get_vector_store().init_collection()


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


class ChatRequest(BaseModel):
    query: str
    category: Optional[str] = None
    session_id: Optional[str] = None


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with RAG and streaming response."""
    retriever = get_retriever()
    llm = get_llm_service()
    
    # Retrieve relevant documents
    results = retriever.retrieve(
        query=request.query,
        category=request.category,
        limit=5
    )
    
    # Build context from retrieved documents
    context_parts = []
    sources = []
    for i, result in enumerate(results, 1):
        payload = result["payload"]
        context_parts.append(f"[{i}] {payload.get('content_chunk', '')}")
        if "url" in payload:
            sources.append(payload["url"])
    
    context = "\n\n".join(context_parts)
    
    # Build prompt
    prompt = f"""You are a knowledgeable assistant specializing in financial and geopolitical news analysis. Use the following context to answer the user's question accurately and concisely.

Context:
{context}

Question: {request.query}

Answer:"""
    
    # Stream response
    async def generate():
        # First, yield sources as JSON
        import json
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        
        # Then stream the LLM response
        for chunk in llm.stream(prompt):
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
