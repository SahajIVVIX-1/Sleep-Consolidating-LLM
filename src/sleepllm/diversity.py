"""Diversity metrics for sets of generated texts. Regulariser comes in a later phase."""
from __future__ import annotations

import math
import random
import re
from collections import Counter
from typing import Callable, Sequence


def _tok(s: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", s.lower())


def distinct_n(texts: Sequence[str], n: int) -> float:
    """Unique n-grams / total n-grams over the whole set (1.0 = nothing repeated)."""
    grams = []
    for t in texts:
        tk = _tok(t)
        grams += [tuple(tk[i:i + n]) for i in range(len(tk) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def _bleu(hyp: list[str], refs: list[list[str]], max_n: int = 4) -> float:
    if not hyp:
        return 0.0
    logp = 0.0
    for n in range(1, max_n + 1):
        h = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
        if not h:
            return 0.0
        mx: Counter = Counter()
        for r in refs:
            for g, c in Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1)).items():
                mx[g] = max(mx[g], c)
        clipped = sum(min(c, mx[g]) for g, c in h.items())
        logp += math.log(max(clipped, 1e-9) / sum(h.values())) / max_n
    ref_len = min((len(r) for r in refs), key=lambda l: (abs(l - len(hyp)), l))
    bp = 1.0 if len(hyp) > ref_len else math.exp(1 - ref_len / len(hyp))
    return bp * math.exp(logp)


def self_bleu(texts: Sequence[str], max_samples: int = 200, seed: int = 0) -> float:
    """Mean BLEU of each text against all the others (high = repetitive)."""
    toks = [_tok(t) for t in texts]
    toks = [t for t in toks if t]
    if len(toks) < 2:
        return 0.0
    idx = list(range(len(toks)))
    rng = random.Random(seed)
    sel = rng.sample(idx, min(max_samples, len(idx)))
    pool = [toks[i] for i in sel]
    return sum(_bleu(h, pool[:i] + pool[i + 1:]) for i, h in enumerate(pool)) / len(pool)


def embedding_dispersion(texts: Sequence[str], embed: Callable[[Sequence[str]], "np.ndarray"] | None = None,
                         max_samples: int = 500, seed: int = 0) -> float:
    """Mean pairwise cosine distance of embeddings (0 = all identical)."""
    import numpy as np
    texts = list(texts)
    if len(texts) < 2:
        return 0.0
    if len(texts) > max_samples:
        texts = random.Random(seed).sample(texts, max_samples)
    E = np.asarray((embed or default_embedder())(texts), dtype=float)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    S = E @ E.T
    n = len(E)
    return float((1 - (S.sum() - np.trace(S)) / (n * (n - 1))))


_EMB = None


def default_embedder():
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        _EMB = lambda xs: m.encode(list(xs), show_progress_bar=False)
    return _EMB


def all_metrics(texts: Sequence[str], embed=None) -> dict:
    return {
        "n": len(texts),
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "distinct_3": distinct_n(texts, 3),
        "self_bleu": self_bleu(texts),
        "embedding_dispersion": embedding_dispersion(texts, embed),
    }
