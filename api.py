"""FastAPI REST API for the adaptive memory system."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.memory_system import MemorySystem

# Global memory instance (in production use dependency injection or app state)
_memory: MemorySystem | None = None


def get_memory() -> MemorySystem:
    if _memory is None:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    return _memory


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _memory
    _memory = MemorySystem()
    yield
    _memory = None


app = FastAPI(title="Adaptive Memory API", lifespan=lifespan)


class AddMemoryRequest(BaseModel):
    content: str


class AddMemoryResponse(BaseModel):
    id: str
    content: str
    weight: float


class QueryResponse(BaseModel):
    memories: list[dict]
    count: int


class DecayResponse(BaseModel):
    hot_count: int
    cold_count: int
    moved_to_cold: int
    updated: int


class StatsResponse(BaseModel):
    hot_count: int
    cold_count: int
    hot_tier_cap: int | None = None


@app.post("/memory/add", response_model=AddMemoryResponse)
def add_memory(body: AddMemoryRequest):
    """Add a new memory."""
    mem = get_memory()
    result = mem.add_memory(body.content)
    return AddMemoryResponse(
        id=result["id"],
        content=result["content"],
        weight=result["weight"],
    )


@app.get("/memory/query", response_model=QueryResponse)
def query_memory(q: str, k: int = 5):
    """Retrieve memories by semantic query."""
    mem = get_memory()
    nodes = mem.retrieve(q, top_k=k)
    return QueryResponse(
        memories=[
            {
                "id": n.id,
                "content": n.content,
                "weight": n.weight,
                "access_count": n.access_count,
            }
            for n in nodes
        ],
        count=len(nodes),
    )


@app.post("/memory/decay", response_model=DecayResponse)
def trigger_decay():
    """Apply time-based decay and move low-weight memories to cold storage."""
    mem = get_memory()
    result = mem.apply_decay()
    return DecayResponse(
        hot_count=result["hot_count"],
        cold_count=result["cold_count"],
        moved_to_cold=result["moved_to_cold"],
        updated=result["updated"],
    )


@app.get("/memory/stats", response_model=StatsResponse)
def memory_stats():
    """Get hot/cold memory counts and hot tier cap."""
    mem = get_memory()
    s = mem.stats()
    return StatsResponse(
        hot_count=s["hot_count"],
        cold_count=s["cold_count"],
        hot_tier_cap=s.get("hot_tier_cap"),
    )


@app.post("/memory/reinforce/{node_id}")
def reinforce_memory(node_id: str):
    """Strengthen a memory by id (e.g. after user finds it relevant)."""
    mem = get_memory()
    ok = mem.reinforce(node_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found in hot storage")
    return {"reinforced": node_id}


@app.get("/health")
def health():
    return {"status": "ok"}
