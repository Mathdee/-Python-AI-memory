`This repository serves as a proof-of-concept for Stigmergic Memory Decay in RAG systems. It is provided as-is for researchers and engineers to fork and adapt.`

# Adaptive Memory System for LLMs

**Bio-inspired decay + tiered storage: optimize forgetting, not just storage.**

## The Problem

Current AI memory systems remember everything equally. That dilutes signal with noise and wastes compute.

## The Solution

- **Single-index architecture**: One HNSW index (Qdrant) stores everything with health metadata. No physical hot/cold movement.
- **Metabolic decay**: Linear health decay (health -= rate × days). Frequent memories (access_count > 2) decay slower. Pinned memories have a health floor (0.2).
- **Health-based filtering**: Dead memories (health < 0.1) filtered during retrieval. No data movement – just metadata updates.
- **Reranking**: `adjusted_score = similarity × (1 + health)`. A 0.8 match with 1.0 health beats a 0.9 match with 0.1 health.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run API
uvicorn api:app --reload --host 0.0.0.0 --port 8000


# Demo (decay + tiering behavior)
python demo.py

# Benchmark (toy: 12 memories)
python tests/benchmark.py

# Scaled benchmark (100–10K memories, 90 days)
python tests/benchmark_scaled.py --quick          # 100 mem, 30 days
python tests/benchmark_scaled.py --scale 1000 --days 90
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/memory/add` | Add a memory (`{"content": "..."}`) |
| GET | `/memory/query?q=...&k=5` | Retrieve by semantic query |
| POST | `/memory/decay` | Apply metabolic decay |
| GET | `/memory/stats` | Hot/cold counts |
| POST | `/memory/reinforce/{id}` | Strengthen a memory by id |

### Moltbot Microservice API (plug-and-play)

When running the Docker container (`src.server:app`), these vector-first endpoints are available. You provide embeddings; the engine handles storage, decay, and retrieval.

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/remember` | Store memory: `{"text": "...", "vector": [384 floats], "metadata": {}}` |
| POST | `/recall` | Retrieve by vector: `{"vector": [384 floats], "top_k": 15}` |
| POST | `/decay?days=1` | Apply metabolic decay |
| GET | `/info` | Vector size (384) and embedding model hint |

## Results Summary

**Toy run (12 memories, 20 days):** On a small dataset, adaptive and full RAG have almost identical latency (~12 ms). Run `python demo.py` and `python tests/benchmark.py` to reproduce.

**Scaled run (1000 memories, 90 days):** Run `python tests/benchmark_scaled.py --quick` or `--scale 1000 --days 90`.

**Benchmark (1K memories, 90 days, daily reinforcement for frequent tier):**

| Metric | Adaptive (1K/90d) | Full RAG |
|--------|-------------------|----------|
| Healthy (above threshold) | 55 items | 1000 items |
| Dead (filtered as noise) | 945 items | 0 |
| Latency (frequent query) | ~14 ms | ~13 ms |
| Latency (noise query) | ~10 ms | ~14 ms |
| Heavy decay (all filtered) | 10.8 ms | 16.5 ms |

- **Noise filtering:** Adaptive filters 945 dead memories (health < 0.1); queries return 0% noise. Full RAG returns noise.
- **Speed when filtered:** When most memories are dead, adaptive is ~35% faster (10.8 ms vs 16.5 ms).
- **Reinforcement matters:** Daily reinforcement keeps ~55 frequent memories hot over 90 days; weekly is too sparse.

**Honest framing:** Early results on 12 memories showed no real advantage. Scaling to 100+ memories and 30+ days is when the pattern emerges: health filtering prunes noise, retrieval stays fast, and adaptive beats full RAG when most memories are dead.

### Tuning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `daily_decay_rate` | 0.05 | 5%/day for unvisited – noise dies in ~20 days. |
| `frequent_decay_rate` | 0.01 | 1%/day for reinforced (access_count > 2) – signal survives. |
| `reinforce_boost` | 0.15 | Linear boost on access: health += boost (cap 1.0). |
| `reinforce_threshold` | 2 | access_count > this → use frequent_decay_rate (protection). |
| `health_floor` | 0.2 | Pinned/core memories never drop below this. |
| `min_health_threshold` | 0.1 | Filter out memories below this (noise filtering). |
| `health_boost_multiplier` | 1.0 | Reranking: adjusted_score = similarity × (1 + health × multiplier). |

**Why this beats the baseline:**

- **Speed**: Single HNSW index (C++ optimized). Metadata filtering is instant. No physical data movement.
- **Recall**: Full dense vectors (no binary quantization). High recall from HNSW, then health-based reranking.
- **Noise filtering**: Dead memories (health < 0.1) filtered out. Precision without sacrificing recall.

## Project Layout

```
Memory Dilution/
├── api.py              # Full FastAPI server (content-based)
├── src/server.py       # Moltbot microservice (vector-first, /remember, /recall)
├── decay_worker.py     # Scheduled decay (e.g. daily 03:00)
├── demo.py             # Demo script
├── requirements.txt
├── src/
│   ├── memory_system.py  # MemorySystem, add_memory_from_vector, retrieve_by_vector
│   └── models.py         # MemoryNode dataclass
└── tests/
    ├── benchmark.py        # Toy benchmark (12 memories)
    └── benchmark_scaled.py # Scaled: 100/1K/10K memories, 90 days
```

## How to Use It

### Method 1: The "Sidecar" (Best for Moltbot)

Run this command to start your memory engine:

```bash
docker build -t metabolic-memory .
docker run -p 8000:8000 metabolic-memory
```

Then, in your agent code, hit the API:

```python
import requests

# Get vector size (384 for all-MiniLM-L6-v2)
r = requests.get("http://localhost:8000/info")
vector_size = r.json()["vector_size"]

# Send a memory (you provide the embedding - use your own model or API)
vector = [...]  # 384-dim embedding of "User likes Python"
requests.post("http://localhost:8000/remember", json={
    "text": "User likes Python",
    "vector": vector,
    "metadata": {}
})

# Recall by query vector
query_vector = [...]  # 384-dim embedding of "What does the user prefer?"
r = requests.post("http://localhost:8000/recall", json={
    "vector": query_vector,
    "top_k": 5
})
memories = r.json()["results"]

# Trigger decay (e.g. daily cron)
requests.post("http://localhost:8000/decay?days=1")
```

### Method 2: The Library (For Builders)

Or, use `src/memory_system.py` directly in your project for a self-contained setup with built-in embeddings:

```python
from src.memory_system import MemorySystem

brain = MemorySystem()
brain.add_memory("User likes Python")  # Auto-embeds with all-MiniLM-L6-v2
results = brain.retrieve("What does the user prefer?", top_k=5)
```

---

## Deploy

**Railway / Render:** Point at `api:app` and set start command to `uvicorn api:app --host 0.0.0.0 --port $PORT`. For the Moltbot microservice, use `src.server:app` instead.

**Docker (Moltbot microservice):**

```bash
docker build -t metabolic-memory .
docker run -p 8000:8000 metabolic-memory

# Quick Deploy
docker-compose up -d
```


## License

[MIT](LICENSE.md)

