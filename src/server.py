"""Moltbot-ready microservice: plug-and-play Metabolic Memory API.

Run with: uvicorn src.server:app --host 0.0.0.0 --port 8000
Docker: docker run -p 8000:8000 metabolic-memory
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

import numpy as np

from .memory_system import MemorySystem, VECTOR_SIZE

app = FastAPI(title="Metabolic Memory Engine")
brain = MemorySystem()


class MemoryItem(BaseModel):
    text: str
    vector: List[float]
    metadata: Dict[str, Any] = {}


class Query(BaseModel):
    vector: List[float]
    top_k: int = 15


@app.post("/remember")
def remember(item: MemoryItem):
    """Store a memory with pre-computed vector. Vector must be length 384 (all-MiniLM-L6-v2)."""
    try:
        brain.add_memory_from_vector(item.text, np.array(item.vector), item.metadata)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "stored", "health": 1.0}


@app.post("/recall")
def recall(query: Query):
    """Retrieve memories by pre-computed query vector."""
    try:
        results = brain.retrieve_by_vector(np.array(query.vector), top_k=query.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Vectors already serialized as lists in retrieve_by_vector
    return {"results": results}


@app.post("/decay")
def trigger_decay(days: int = 1):
    """Apply metabolic decay for the given number of days. Returns count of memories below health threshold."""
    result = brain.apply_decay(simulate_days_elapsed=days)
    memories_pruned = result.get("below_threshold", 0)
    return {"status": "metabolism_complete", "memories_pruned": memories_pruned}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    """Return vector size and other info for client embedding compatibility."""
    return {"vector_size": VECTOR_SIZE, "embedding_model": "all-MiniLM-L6-v2"}
