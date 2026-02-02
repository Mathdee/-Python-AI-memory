"""Demo script: show decay and tiering behavior."""
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.memory_system import MemorySystem


def main():
    print("=== Adaptive Memory Demo ===\n")

    memory = MemorySystem(decay_rate=0.95, cold_threshold=0.3)

    # 1. Chat about Python (frequent)
    print("1. Adding 10 messages about Python...")
    for i in range(10):
        memory.add_memory(
            f"User: What's recursion? Assistant: Recursion is when a function calls itself. Example: factorial(n) = n * factorial(n-1)."
        )
    print(f"   Stats: {memory.stats()}\n")

    # 2. Chat about cooking (few)
    print("2. Adding 2 messages about cooking...")
    memory.add_memory("User: Best pasta recipe? Assistant: Boil water, salt, add pasta, drain, add sauce.")
    memory.add_memory("User: How long to boil? Assistant: About 10-12 minutes for spaghetti.")
    print(f"   Stats: {memory.stats()}\n")

    # 3. Query before decay
    print("3. Query before decay:")
    for q in ["What did we discuss about recursion?", "What did we discuss about cooking?"]:
        results = memory.retrieve(q, top_k=3)
        print(f"   Q: {q}")
        print(f"   -> {len(results)} results. First: {results[0].content[:80] if results else 'none'}...")
    print()

    # 4. Simulate time: apply decay with simulated days (so weights actually drop)
    print("4. Applying decay with simulated 20 days (reinforcing Python only)...")
    for _ in range(20):
        memory.apply_decay(simulate_days_elapsed=1)
        # Reinforce Python (frequently used) so it stays hot
        memory.retrieve("recursion", top_k=2)
    print(f"   Stats after decay: {memory.stats()}\n")

    # 5. Query after decay
    print("5. Query after decay:")
    for q in ["What did we discuss about recursion?", "What did we discuss about cooking?"]:
        results = memory.retrieve(q, top_k=3)
        print(f"   Q: {q}")
        if results:
            print(f"   -> {len(results)} results. First: {results[0].content[:80]}... (weight={results[0].weight:.2f})")
        else:
            print("   -> No results (likely in cold or pruned).")
    print("\nDone. Python (reinforced) stays hot; cooking (rare) decays to cold.")


if __name__ == "__main__":
    main()
