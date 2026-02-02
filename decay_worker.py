"""Background decay worker: runs apply_decay on a schedule."""
from __future__ import annotations

import schedule
import time
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.memory_system import MemorySystem

# Shared in-memory system (in production, attach to same DB or use API)
memory = MemorySystem()


def scheduled_decay():
    result = memory.apply_decay()
    hot = result["hot_count"]
    cold = result["cold_count"]
    moved = result["moved_to_cold"]
    print(f"Decay applied. Hot: {hot}, Cold: {cold}, Moved to cold: {moved}")


def main():
    # Run decay every 24 hours at 03:00
    schedule.every().day.at("03:00").do(scheduled_decay)
    # For testing: also run every minute
    # schedule.every(1).minutes.do(scheduled_decay)

    print("Decay worker started. Next decay at 03:00 daily. Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(3600)


if __name__ == "__main__":
    main()
