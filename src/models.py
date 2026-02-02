"""Data models for the adaptive memory system."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class MemoryNode:
    """A single memory unit with embedding and decay metadata."""
    id: str
    content: str
    embedding: np.ndarray
    weight: float
    timestamp: datetime
    access_count: int = 0
    last_accessed: datetime | None = None

    def to_payload(self) -> dict[str, Any]:
        """Serialize for Qdrant payload (no numpy)."""
        out = {
            "content": self.content,
            "weight": float(self.weight),
            "timestamp": self.timestamp.isoformat(),
            "access_count": self.access_count,
        }
        if self.last_accessed is not None:
            out["last_accessed"] = self.last_accessed.isoformat()
        return out

    @classmethod
    def from_payload(cls, id: str, vector: list[float], payload: dict) -> "MemoryNode":
        """Reconstruct from Qdrant point."""
        ts = datetime.fromisoformat(payload["timestamp"])
        last = payload.get("last_accessed")
        last_accessed = datetime.fromisoformat(last) if last else None
        return cls(
            id=id,
            content=payload["content"],
            embedding=np.array(vector, dtype=np.float32),
            weight=float(payload["weight"]),
            timestamp=ts,
            access_count=int(payload.get("access_count", 0)),
            last_accessed=last_accessed,
        )
