"""Adaptive memory system: two-tier storage, decay, and radio-signal retrieval."""
from __future__ import annotations

import math
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer

from .models import MemoryNode

COLLECTION_NAME = "hot_memories"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 default

# Metabolic memory params: "Aggressive decay, strong reinforcement"
DAILY_DECAY_RATE = 0.05  # 5%/day for unvisited – noise dies in ~20 days
FREQUENT_DECAY_RATE = 0.01  # 1%/day for reinforced (access_count > 2) – signal survives
REINFORCE_BOOST = 0.15  # Strong boost on access: health += boost (cap at 1.0)
HEALTH_FLOOR = 0.2  # Core/pinned memories never fully die
MIN_HEALTH_THRESHOLD = 0.1  # Filter out memories below this (noise filtering)
REINFORCE_THRESHOLD = 2  # access_count > this → use frequent_decay_rate (protection)
HEALTH_BOOST_MULTIPLIER = 1.0  # Reranking: adjusted_score = similarity × (1 + health × multiplier)

# Common English stopwords (minimal set for keyword extraction)
_STOPWORDS = frozenset(
    "a an the and or but in on at to for of with by from as is was are were been be have has had do does did will would could should may might must can shall".split()
)


def _extract_keywords(content: str) -> list[str]:
    """Extract keywords from content via simple tokenization (no LLM)."""
    text = content.lower().strip()
    tokens = re.findall(r"\b[a-z0-9]{2,}\b", text)
    return [t for t in tokens if t not in _STOPWORDS]


def _summarize(content: str, max_chars: int = 200) -> str:
    """Summarize content for cold storage (simple truncation; replace with LLM if needed)."""
    content = content.strip()
    if len(content) <= max_chars:
        return content
    return content[: max_chars - 3].rsplit(maxsplit=1)[0] + "..."


class MemorySystem:
    """Single-index metabolic memory: HNSW + health-based filtering. No physical hot/cold movement."""

    def __init__(
        self,
        daily_decay_rate: float = DAILY_DECAY_RATE,
        frequent_decay_rate: float = FREQUENT_DECAY_RATE,
        reinforce_boost: float = REINFORCE_BOOST,
        health_floor: float = HEALTH_FLOOR,
        min_health_threshold: float = MIN_HEALTH_THRESHOLD,
        health_boost_multiplier: float = HEALTH_BOOST_MULTIPLIER,
        persist_path: str | Path | None = None,
        # Legacy params (backward compat - map to new system)
        decay_rate: float | None = None,
        hot_tier_cap: int | None = None,
        cold_threshold: float | None = None,
        reinforce_factor: float | None = None,
        resonance_threshold: float | None = None,
        hot_search_limit: int | None = None,
        cold_search_limit: int | None = None,
        cold_vector_weight: float | None = None,
        cold_keyword_weight: float | None = None,
        use_access_based_decay: bool = True,
        confidence_threshold: float = 0.6,
        hybrid_cold: bool = True,
        min_cold_score: float = 0.35,
        daily_decay_factor: float | None = None,
    ):
        self.daily_decay_rate = daily_decay_rate
        self.frequent_decay_rate = frequent_decay_rate
        self.reinforce_boost = reinforce_boost
        self.health_floor = health_floor
        self.min_health_threshold = min_health_threshold
        self.health_boost_multiplier = health_boost_multiplier
        self.reinforce_threshold = 2  # access_count > this → use frequent_decay_rate
        self.persist_path = Path(persist_path) if persist_path else None

        # Backward compat: legacy decay_rate ignored – use explicit daily_decay_rate

        # Single HNSW index (Qdrant) - everything stored here with health metadata
        self._client = QdrantClient(":memory:")
        self._ensure_collection()

        # Embedding model (lazy load)
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def _ensure_collection(self) -> None:
        collections = self._client.get_collections().collections
        names = [c.name for c in collections]
        if COLLECTION_NAME not in names:
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(
                    size=VECTOR_SIZE, distance=qdrant_models.Distance.COSINE
                ),
            )

    def _embed(self, text: str) -> np.ndarray:
        model = self._get_model()
        return model.encode(text, convert_to_numpy=True).astype(np.float32)

    def _count(self) -> int:
        """Total memories in the index."""
        try:
            return self._client.get_collection(COLLECTION_NAME).points_count
        except Exception:
            return 0

    def add_memory_from_vector(
        self,
        content: str,
        vector: np.ndarray,
        metadata: dict[str, Any] | None = None,
        is_pinned: bool = False,
    ) -> dict[str, Any]:
        """
        Add memory with pre-computed vector (Moltbot/plug-and-play API).
        Vector must match VECTOR_SIZE (384 for all-MiniLM-L6-v2).
        """
        vec = np.asarray(vector, dtype=np.float32)
        if vec.shape != (VECTOR_SIZE,):
            raise ValueError(f"Vector must be length {VECTOR_SIZE}, got {vec.shape[0]}")
        node_id = str(uuid.uuid4())
        now = datetime.utcnow()
        payload = {
            "content": content,
            "health": 1.0,
            "timestamp": now.isoformat(),
            "access_count": 0,
            "last_accessed": now.isoformat(),
            "is_pinned": is_pinned,
            **(metadata or {}),
        }
        self._client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                qdrant_models.PointStruct(
                    id=node_id,
                    vector=vec.tolist(),
                    payload=payload,
                )
            ],
        )
        return {"id": node_id, "content": content[:100], "health": 1.0, "weight": 1.0}

    def add_memory(self, content: str, is_pinned: bool = False) -> dict[str, Any]:
        """Add memory to single index with health=1.0. No eviction – health filtering handles noise."""
        node_id = str(uuid.uuid4())
        embedding = self._embed(content)
        now = datetime.utcnow()
        
        payload = {
            "content": content,
            "health": 1.0,  # Start at full health
            "timestamp": now.isoformat(),
            "access_count": 0,
            "last_accessed": now.isoformat(),
            "is_pinned": is_pinned,  # Pinned memories have health floor
        }
        
        self._client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                qdrant_models.PointStruct(
                    id=node_id,
                    vector=embedding.tolist(),
                    payload=payload,
                )
            ],
        )
        return {"id": node_id, "content": content[:100], "health": 1.0, "weight": 1.0}

    def get_node(self, node_id: str) -> MemoryNode | None:
        """Fetch a single node by id."""
        try:
            points = self._client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[node_id],
                with_payload=True,
                with_vectors=True,
            )
        except Exception:
            points = []
        if points:
            p = points[0]
            vec = p.vector
            if isinstance(vec, dict):
                vec = next(iter(vec.values()), []) if vec else []
            if vec is None:
                vec = []
            payload = p.payload or {}
            return MemoryNode(
                id=str(p.id),
                content=payload.get("content", ""),
                embedding=np.array(vec, dtype=np.float32),
                weight=payload.get("health", 0.0),  # Map health → weight for compat
                timestamp=datetime.fromisoformat(
                    payload.get("timestamp", datetime.utcnow().isoformat())
                ),
                access_count=payload.get("access_count", 0),
            )
        return None

    def reinforce(self, node_id: str) -> bool:
        """Boost health on access. Linear: health += boost (cap at 1.0)."""
        try:
            points = self._client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[node_id],
                with_payload=True,
                with_vectors=True,
            )
        except Exception:
            return False
        if not points:
            return False
        
        p = points[0]
        payload = p.payload or {}
        vec = p.vector
        if isinstance(vec, dict):
            vec = next(iter(vec.values()), []) if vec else []
        if vec is None:
            vec = []
        
        # Linear boost
        payload["health"] = min(1.0, payload.get("health", 0.0) + self.reinforce_boost)
        payload["access_count"] = payload.get("access_count", 0) + 1
        payload["last_accessed"] = datetime.utcnow().isoformat()
        
        self._client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                qdrant_models.PointStruct(
                    id=node_id,
                    vector=vec if isinstance(vec, list) else vec.tolist(),
                    payload=payload,
                )
            ],
        )
        return True

    def _retrieve_core(
        self, query_vector: list[float], top_k: int
    ) -> list[tuple[MemoryNode, float]]:
        """Shared vector search + filter + rerank. Returns (node, adjusted_score) pairs."""
        response = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k * 3,
            with_payload=True,
            with_vectors=True,
            score_threshold=None,
        )
        hits = response.points if hasattr(response, "points") else []
        candidates: list[tuple[MemoryNode, float]] = []
        for hit in hits:
            payload = hit.payload or {}
            health = payload.get("health", 0.0)
            if health < self.min_health_threshold:
                continue
            similarity = getattr(hit, "score", 0.0)
            vec = hit.vector
            if isinstance(vec, dict):
                vec = next(iter(vec.values()), []) if vec else []
            if vec is None:
                vec = []
            adjusted_score = similarity * (1 + health * self.health_boost_multiplier)
            node = MemoryNode(
                id=str(hit.id),
                content=payload.get("content", ""),
                embedding=np.array(vec, dtype=np.float32),
                weight=health,
                timestamp=datetime.fromisoformat(
                    payload.get("timestamp", datetime.utcnow().isoformat())
                ),
                access_count=payload.get("access_count", 0),
            )
            candidates.append((node, adjusted_score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def retrieve_by_vector(
        self, query_vector: np.ndarray, top_k: int = 15
    ) -> list[dict[str, Any]]:
        """
        Metabolic retrieval by pre-computed vector (Moltbot/plug-and-play API).
        Returns list of dicts: id, content, vector, health, metadata, etc.
        """
        vec = np.asarray(query_vector, dtype=np.float32)
        if vec.shape != (VECTOR_SIZE,):
            raise ValueError(f"Vector must be length {VECTOR_SIZE}, got {vec.shape[0]}")
        pairs = self._retrieve_core(vec.tolist(), top_k)
        results: list[dict[str, Any]] = []
        for node, _ in pairs:
            self.reinforce(node.id)
            results.append({
                "id": node.id,
                "content": node.content,
                "vector": node.embedding.tolist(),
                "health": node.weight,
                "access_count": node.access_count,
            })
        return results

    def retrieve(self, query: str, top_k: int = 15) -> list[MemoryNode]:
        """
        Metabolic retrieval: HNSW search + health-based reranking.

        1. Vector search (high recall – search everything)
        2. Filter out dead memories (health < min_threshold)
        3. Rerank: adjusted_score = similarity × (1 + health × multiplier)
        4. Reinforce accessed memories
        """
        query_vector = self._embed(query)
        pairs = self._retrieve_core(query_vector.tolist(), top_k)
        results = [node for node, _ in pairs]
        for node in results:
            self.reinforce(node.id)
        return results

    def apply_decay(self, simulate_days_elapsed: int | None = None) -> dict[str, int]:
        """
        Aggressive decay for unvisited, protection for reinforced.
        
        - Unvisited (access_count <= 2): 5%/day → dead in ~20 days
        - Reinforced (access_count > 2): 1%/day → signal survives 90 days
        - Pinned: never drop below health_floor
        - Dead (health < 0.1): filtered at query time
        """
        days = 1 if simulate_days_elapsed is None else max(1, simulate_days_elapsed)
        updated = 0
        below_threshold = 0
        offset = None
        reinforce_threshold = getattr(self, "reinforce_threshold", 2)

        while True:
            records, offset = self._client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not records:
                break
            
            for r in records:
                payload = r.payload or {}
                health = payload.get("health", 0.0)
                access_count = payload.get("access_count", 0)
                is_pinned = payload.get("is_pinned", False)
                
                # Aggressive decay for unvisited, protection for reinforced
                current_decay = self.frequent_decay_rate if access_count > reinforce_threshold else self.daily_decay_rate
                
                health -= (current_decay * days)
                
                # Floor: pinned = 0.2, else 0 (allow full death for filtering)
                if is_pinned:
                    health = max(health, self.health_floor)
                else:
                    health = max(health, 0.0)
                
                payload["health"] = health
                
                # Count how many are below threshold (for stats)
                if health < self.min_health_threshold:
                    below_threshold += 1
                
                vec = r.vector
                if isinstance(vec, dict):
                    vec = next(iter(vec.values()), []) if vec else []
                if vec is None:
                    vec = []
                
                self._client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[
                        qdrant_models.PointStruct(
                            id=str(r.id),
                            vector=vec if isinstance(vec, list) else vec.tolist(),
                            payload=payload,
                        )
                    ],
                )
                updated += 1
            
            if offset is None:
                break

        total_count = self._count()
        healthy_count = total_count - below_threshold

        # Backward compat for API/decay_worker (hot=healthy, cold=dead, no physical movement)
        return {
            "total_count": total_count,
            "healthy_count": healthy_count,
            "below_threshold": below_threshold,
            "updated": updated,
            "hot_count": healthy_count,
            "cold_count": below_threshold,
            "moved_to_cold": 0,
        }

    def stats(self) -> dict[str, Any]:
        """Return memory stats: total, healthy (above threshold), dead (below threshold)."""
        total = self._count()
        healthy = 0
        dead = 0
        
        try:
            records, _ = self._client.scroll(
                collection_name=COLLECTION_NAME,
                limit=total,
                with_payload=True,
            )
            for r in records:
                health = (r.payload or {}).get("health", 0.0)
                if health >= self.min_health_threshold:
                    healthy += 1
                else:
                    dead += 1
        except Exception:
            pass
        
        return {
            "total_count": total,
            "healthy_count": healthy,
            "dead_count": dead,
            "min_health_threshold": self.min_health_threshold,
            # Backward compat
            "hot_count": healthy,
            "cold_count": dead,
        }
