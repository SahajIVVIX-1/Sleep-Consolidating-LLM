"""Old-skill replay buffer. STUB in Phase 1: returns nothing, so Phase 1 measures unprotected forgetting."""
from __future__ import annotations


class ReplayBuffer:
    def __init__(self, items=None):
        self.items = list(items or [])

    def sample(self, n: int) -> list[dict]:
        return []  # Phase 2: return n {"user","assistant"} items from self.items
