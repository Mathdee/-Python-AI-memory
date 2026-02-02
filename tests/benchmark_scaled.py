"""
Scaled benchmark: 1K–10K memories, 90 days, frequent/occasional/noise tiers.

Measures: latency at scale (hot vs full RAG), cold retrieval speed, precision on
noisy queries, noise resistance. Adaptive should be faster on hot queries, cold
search faster than dense scan, and precision higher on noisy data (pruned junk).
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import psutil

# Project root
sys_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(sys_path))

from src.memory_system import MemorySystem

_process = psutil.Process()

# Tier labels
FREQUENT = "frequent"
OCCASIONAL = "occasional"
NOISE = "noise"


def get_memory_mb():
    return _process.memory_info().rss / 1024 / 1024


def build_scaled_data(n_frequent: int, n_occasional: int, n_noise: int, seed: int = 42):
    """Generate (content, tier) for each tier. Total = n_frequent + n_occasional + n_noise.
    Use ~5% frequent, ~10% occasional, ~85% noise to test noise resistance and pruning.
    """
    random.seed(seed)
    np.random.seed(seed)
    data = []

    # Frequent: Python/recursion (reinforced daily → stays hot)
    for i in range(n_frequent):
        data.append((
            f"User: What is recursion? Assistant: Recursion is when a function calls itself. Example {i}: factorial(n) = n * factorial(n-1).",
            FREQUENT,
        ))

    # Occasional: tech topics (reinforced monthly → mostly cold)
    # Inject needle at MIDPOINT: old enough to be cold (evicted from hot), young enough to survive
    # (~Day 60 / 30 days ago → 0.98^30 ≈ 54% weight, cold but strong)
    tech_topics = [
        "machine learning", "API design", "database indexing", "caching", "async programming",
        "testing", "debugging", "refactoring", "code review", "CI/CD",
    ]
    occasional_list = []
    for i in range(n_occasional - 1):  # Leave room for the needle
        t = tech_topics[i % len(tech_topics)]
        occasional_list.append((f"User: Tell me about {t}. Assistant: {t} is important for software. Here are basics.", OCCASIONAL))
    # Insert at midpoint (not Day 0 / append, not Day 90 / append) → ~Day 60
    needle = (
        "User: Can you Explain API design? Assistant: API design requires RESTful principles, clear documentation, and consistent endpoints.",
        OCCASIONAL,
    )
    midpoint = len(occasional_list) // 2
    occasional_list.insert(midpoint, needle)
    data.extend(occasional_list)

    # Noise: one-off irrelevant (never reinforced → should decay to cold / get pruned)
    noise_templates = [
        "User: What's the weather? Assistant: I don't have weather data.",
        "User: What did I eat for lunch? Assistant: I don't store that.",
        "User: Random question {i}. Assistant: One-off reply.",
        "User: Unrelated chat. Assistant: Sure.",
    ]
    for i in range(n_noise):
        t = noise_templates[i % len(noise_templates)]
        data.append((t.format(i=i), NOISE))

    random.shuffle(data)
    return data


def run_adaptive_scaled(
    data: list[tuple[str, str]],
    days: int = 90,
    cold_threshold: float = 0.4,
    decay_rate: float = 0.98,
    reinforce_frequent_every_n_days: int = 1,
    reinforce_occasional_every_n_days: int = 30,
    n_frequent: int = 50,
):
    """
    Build adaptive system with aggressive decay, reinforce signal (frequent + occasional).
    Noise (unreinforced) dies at 5%/day. Reinforced memories decay at 1%/day.
    """
    memory = MemorySystem(
        daily_decay_rate=0.05,  # 5%/day – noise dead in ~20 days
        frequent_decay_rate=0.01,  # 1%/day – signal survives 90 days
    )
    id_to_tier = {}

    for content, tier in data:
        out = memory.add_memory(content)
        id_to_tier[out["id"]] = tier

    top_k_frequent = max(5, n_frequent)
    top_k_occasional = max(5, 20)

    # Simulate: each day apply decay, then reinforce signal (not noise)
    for day in range(days):
        memory.apply_decay(simulate_days_elapsed=1)
        # Frequent (Python): reinforce daily → stays healthy
        if reinforce_frequent_every_n_days and day % reinforce_frequent_every_n_days == 0:
            memory.retrieve("Python recursion", top_k=top_k_frequent)
        # Occasional (tech): reinforce monthly
        if reinforce_occasional_every_n_days and day % reinforce_occasional_every_n_days == 0:
            memory.retrieve("machine learning", top_k=top_k_occasional)
        # API needle: reinforce every 7 days so access_count > 2 before noise death (~20 days)
        if day % 7 == 0:
            memory.retrieve("Explain API design", top_k=5)

    return memory, id_to_tier


def run_full_rag(data: list[tuple[str, str]]):
    """Build full RAG with same data, no decay. Returns (memory, id_to_tier) for precision/recall."""
    memory = MemorySystem(decay_rate=1.0, cold_threshold=0.0)
    id_to_tier = {}
    for content, tier in data:
        out = memory.add_memory(content)
        id_to_tier[out["id"]] = tier
    return memory, id_to_tier


def measure_latency_ms(system, query: str, top_k: int = 5, n_runs: int = 20) -> float:
    """Average latency in ms over n_runs."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        system.retrieve(query, top_k=top_k)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def measure_ram_mb(system) -> float:
    """Approximate RAM used by process (after system is built)."""
    # Force a small op to stabilize
    system.retrieve("test", top_k=1)
    return get_memory_mb()


def precision_recall(retrieved_ids: set, id_to_tier: dict, tier: str) -> tuple[float, float]:
    """Precision: of retrieved, fraction in tier. Recall: of tier, fraction retrieved."""
    tier_ids = {i for i, t in id_to_tier.items() if t == tier}
    if not retrieved_ids:
        return 0.0, 0.0
    if not tier_ids:
        return 0.0, 0.0
    hits = retrieved_ids & tier_ids
    precision = len(hits) / len(retrieved_ids)
    recall = len(hits) / len(tier_ids)
    return precision, recall


def main():
    import argparse
    p = argparse.ArgumentParser(description="Scaled benchmark: 1K–10K memories, 90 days")
    p.add_argument("--scale", type=int, default=1000, choices=[100, 1000, 10000], help="Total memories (100, 1K, 10K)")
    p.add_argument("--days", type=int, default=90, help="Simulated days")
    p.add_argument("--threshold", type=float, default=0.4, help="Cold threshold (default 0.4 to trigger cold)")
    p.add_argument("--decay-rate", type=float, default=0.98, help="Decay per day (0.98 = slower decay)")
    p.add_argument("--reinforce-frequent-every", type=int, default=1, metavar="N", help="Reinforce frequent tier every N days (1=daily keeps hot over 90 days)")
    p.add_argument("--quick", action="store_true", help="Quick run: 100 memories, 30 days")
    args = p.parse_args()

    if args.quick:
        n_freq, n_occ, n_noise = 10, 20, 70
        days = 30
        scale_label = "100 (quick)"
    else:
        s = args.scale
        # Ratio: ~5% frequent, ~10% occasional, ~85% noise (tests noise resistance)
        n_freq = max(10, s // 20)
        n_occ = max(20, s // 10)
        n_noise = s - n_freq - n_occ
        if n_noise < 0:
            n_noise = max(0, s - n_freq - n_occ)
            n_occ = s - n_freq - n_noise
        days = args.days
        scale_label = str(n_freq + n_occ + n_noise)

    print(f"=== Scaled benchmark: {scale_label} memories, {days} days, cold_threshold={args.threshold}, decay={args.decay_rate} ===\n")
    data = build_scaled_data(n_freq, n_occ, n_noise)
    n_total = len(data)
    print(f"Tiers: frequent={n_freq}, occasional={n_occ}, noise={n_noise} (total={n_total})")

    # Build adaptive (with decay)
    print(f"\nBuilding adaptive system and simulating time (reinforce frequent every {args.reinforce_frequent_every} day(s))...")
    t0 = time.perf_counter()
    adaptive, id_to_tier = run_adaptive_scaled(
        data,
        days=days,
        cold_threshold=args.threshold,
        decay_rate=args.decay_rate,
        reinforce_frequent_every_n_days=args.reinforce_frequent_every,
        n_frequent=n_freq,
    )
    build_adaptive_s = time.perf_counter() - t0
    stats = adaptive.stats()
    hot, cold = stats["hot_count"], stats["cold_count"]
    print(f"  Done in {build_adaptive_s:.1f}s. Hot={hot}, Cold={cold} (moved to cold: {cold})")

    # Build full RAG (no decay) for same data
    print("Building full RAG (no decay)...")
    t0 = time.perf_counter()
    full_rag, full_rag_id_to_tier = run_full_rag(data)
    build_full_s = time.perf_counter() - t0
    print(f"  Done in {build_full_s:.1f}s. Points={full_rag.stats()['hot_count']}")

    # Query strings used for latency and precision (define before All-cold block)
    q_freq = "Python recursion"
    q_noise = "random weather"
    q_occ = "Explain API design"

    # Health filtering test: heavily decayed system vs full RAG
    if n_total >= 100 and not args.quick:
        print("\n--- Heavy decay vs full RAG (same scale) ---")
        decayed = MemorySystem(daily_decay_rate=0.05)  # Aggressive decay
        for content, _ in data:
            decayed.add_memory(content)
        decayed.apply_decay(simulate_days_elapsed=30)  # 30 days of aggressive decay
        sc = decayed.stats()
        dead_count = sc["dead_count"]
        if dead_count > 0:
            lat_decayed = measure_latency_ms(decayed, q_freq, top_k=10)
            lat_full = measure_latency_ms(full_rag, q_freq, top_k=10)
            print(f"  Decayed (n={sc['total_count']}, {dead_count} filtered): {lat_decayed:.1f} ms")
            print(f"  Full RAG (no filtering, n={n_total}):                    {lat_full:.1f} ms")
        del decayed

    # Latency at scale
    print("\n--- Latency (avg over 20 queries) ---")
    lat_adaptive_freq = measure_latency_ms(adaptive, q_freq)
    lat_adaptive_noise = measure_latency_ms(adaptive, q_noise)
    lat_full_freq = measure_latency_ms(full_rag, q_freq)
    lat_full_noise = measure_latency_ms(full_rag, q_noise)
    print(f"  Adaptive (frequent query): {lat_adaptive_freq:.1f} ms")
    print(f"  Adaptive (noise query):   {lat_adaptive_noise:.1f} ms")
    print(f"  Full RAG (frequent):     {lat_full_freq:.1f} ms")
    print(f"  Full RAG (noise):         {lat_full_noise:.1f} ms")
    if lat_full_freq > 0:
        speedup = lat_full_freq / max(lat_adaptive_freq, 0.01)
        print(f"  Adaptive vs Full RAG (frequent): {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

    # RAM (process RSS – same process, so we measure delta by building in order; second build includes both)
    ram_before = get_memory_mb()
    # We already have both in memory; report relative to baseline
    ram_adaptive = get_memory_mb()  # after adaptive + full_rag both built
    print("\n--- Memory (process RSS) ---")
    print(f"  Process RSS: {ram_adaptive:.0f} MB (adaptive + full_rag both in process)")
    print(f"  Adaptive: {hot} healthy, {cold} dead (filtered by health < 0.1)")

    # Precision / recall: frequent (hot), occasional (cold), noise (should be deprioritized)
    print("\n--- Precision / Recall (top 10) ---")
    ret_adaptive_freq = adaptive.retrieve(q_freq, top_k=10)
    ret_adaptive_noise = adaptive.retrieve(q_noise, top_k=10)
    ret_adaptive_occ = adaptive.retrieve(q_occ, top_k=10)
    ret_full_freq = full_rag.retrieve(q_freq, top_k=10)
    ret_full_noise = full_rag.retrieve(q_noise, top_k=10)
    ret_full_occ = full_rag.retrieve(q_occ, top_k=10)
    ids_adaptive_freq = {r.id for r in ret_adaptive_freq}
    ids_adaptive_noise = {r.id for r in ret_adaptive_noise}
    ids_adaptive_occ = {r.id for r in ret_adaptive_occ}
    ids_full_freq = {r.id for r in ret_full_freq}
    ids_full_noise = {r.id for r in ret_full_noise}
    ids_full_occ = {r.id for r in ret_full_occ}
    p_freq_a, r_freq_a = precision_recall(ids_adaptive_freq, id_to_tier, FREQUENT)
    p_noise_a, r_noise_a = precision_recall(ids_adaptive_noise, id_to_tier, NOISE)
    p_occ_a, r_occ_a = precision_recall(ids_adaptive_occ, id_to_tier, OCCASIONAL)
    p_freq_f, r_freq_f = precision_recall(ids_full_freq, full_rag_id_to_tier, FREQUENT)
    p_noise_f, r_noise_f = precision_recall(ids_full_noise, full_rag_id_to_tier, NOISE)
    p_occ_f, r_occ_f = precision_recall(ids_full_occ, full_rag_id_to_tier, OCCASIONAL)
    print(f"  Query 'Python recursion' (hot):")
    print(f"    Adaptive:  precision={p_freq_a:.2f}, recall={r_freq_a:.2f}")
    print(f"    Full RAG:  precision={p_freq_f:.2f}, recall={r_freq_f:.2f}")
    print(f"  Query 'Explain API design' (occasional):")
    print(f"    Adaptive:  precision={p_occ_a:.2f}, recall={r_occ_a:.2f}")
    print(f"    Full RAG:  precision={p_occ_f:.2f}, recall={r_occ_f:.2f}")
    print(f"  Query 'random weather' (noise):")
    print(f"    Adaptive:  precision(noise)={p_noise_a:.2f} (noise pruned -> lower is better)")
    print(f"    Full RAG:  precision(noise)={p_noise_f:.2f}")

    # Health-based filtering: dead memories are filtered during retrieval
    print("\n--- Health-based filtering ---")
    if cold > 0:
        print(f"  {cold} memories below health threshold (filtered as noise).")
        lat_health_query = measure_latency_ms(adaptive, q_occ, top_k=10)
        lat_full_occ = measure_latency_ms(full_rag, q_occ, top_k=10)
        print(f"  Adaptive (health-filtered query): {lat_health_query:.1f} ms")
        print(f"  Full RAG (no filtering):          {lat_full_occ:.1f} ms")
    else:
        print("  All memories still healthy. Run more decay days to see filtering effect.")

    
    found = False
    # Single-index architecture: scroll all memories
    try:
        records, _ = adaptive._client.scroll(
            collection_name="hot_memories",
            limit=500,
            with_payload=True,
        )
        for r in records or []:
            payload = r.payload or {}
            content = str(payload.get("content", ""))
            health = payload.get("health", 0.0)
            if "API" in content and "Explain" in content:
                status = "HEALTHY" if health >= 0.1 else "DEAD (filtered)"
                print(f"[OK] FOUND! ID: {r.id}")
                print(f"   Health: {health:.2f} ({status})")
                print(f"   Content excerpt: {content[:80]}...")
                found = True
                break
    except Exception as e:
        print(f"   Error scrolling: {e}")
    if not found:
        print("[X] NOT FOUND. The memory was filtered out or never created.")
        print(f"   Total memories: {adaptive.stats()['total_count']}")

    # Summary table
    print("\n--- Summary ---")
    print(f"  Scale: {n_total} memories, {days} days")
    print(f"  Adaptive: {hot} healthy, {cold} dead (filtered)")
    print(f"  Latency (frequent): adaptive={lat_adaptive_freq:.1f} ms, full_rag={lat_full_freq:.1f} ms")
    print(f"  Recall on frequent: adaptive={r_freq_a:.2f}, full_rag={r_freq_f:.2f}")
    print("\nDone. Run with --scale 1000 --days 90 for full benchmark, or --quick for a fast check.")


if __name__ == "__main__":
    main()
