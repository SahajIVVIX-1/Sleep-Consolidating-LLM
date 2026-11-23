"""Day memory: append-only JSONL notes, keyword (BM25) retrieval into the prompt.

A note may carry (subject, relation) keys. When two notes share keys, only the
latest (highest seq) is visible to retrieval (updates supersede earlier facts).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .utils import append_jsonl, read_jsonl


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9\-]+", s.lower())


@dataclass
class Note:
    seq: int
    day: int
    text: str
    subject: str | None = None
    relation: str | None = None
    object: str | None = None


class Notebook:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.notes: list[Note] = [Note(**r) for r in read_jsonl(self.path)]

    def upto(self, day: int) -> "Notebook":
        """In-memory view containing only notes written on or before `day` (no file attached)."""
        v = Notebook.__new__(Notebook)
        v.path, v.notes = None, [n for n in self.notes if n.day <= day]
        return v

    def append(self, day: int, text: str, subject=None, relation=None, object=None) -> Note:
        n = Note(len(self.notes), day, text, subject, relation, object)
        self.notes.append(n)
        append_jsonl(self.path, asdict(n))
        return n

    def read(self, day: int | None = None, active_only: bool = False) -> list[Note]:
        notes = self.active() if active_only else self.notes
        return [n for n in notes if day is None or n.day == day]

    def active(self) -> list[Note]:
        latest: dict[tuple[str, str], int] = {}
        for n in self.notes:
            if n.subject and n.relation:
                latest[(n.subject.lower(), n.relation)] = n.seq
        return [n for n in self.notes
                if not (n.subject and n.relation) or latest[(n.subject.lower(), n.relation)] == n.seq]

    def retrieve(self, query: str, k: int = 5) -> list[Note]:
        docs = self.active()
        if not docs:
            return []
        toks = [_tok(n.text) for n in docs]
        N = len(docs)
        df = Counter(t for ts in toks for t in set(ts))
        avg = sum(len(t) for t in toks) / N
        q = _tok(query)
        scored = []
        for n, ts in zip(docs, toks):
            tf = Counter(ts)
            s = 0.0
            for t in set(q):
                if t in tf:
                    idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(ts) / avg))
            scored.append((s, n.seq, n))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [n for s, _, n in scored[:k] if s > 0]

    def format_prompt(self, query: str, k: int = 5) -> str:
        return "\n".join(f"- {n.text}" for n in self.retrieve(query, k))
