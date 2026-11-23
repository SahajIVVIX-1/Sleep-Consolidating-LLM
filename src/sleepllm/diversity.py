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


# ---------------------------------------------------------------- Phase 2: vocab entropy, regulariser
def vocab_entropy(texts: Sequence[str]) -> float:
    """Shannon entropy (bits) of the unigram distribution."""
    c = Counter(w for t in texts for w in _tok(t))
    tot = sum(c.values())
    return -sum(v / tot * math.log2(v / tot) for v in c.values()) if tot else 0.0


def night_metrics(dream_texts: Sequence[str], note_texts: Sequence[str], embed=None) -> dict:
    m = all_metrics(dream_texts, embed)
    m["vocab_entropy"] = vocab_entropy(dream_texts)
    m["notes_vocab_entropy"] = vocab_entropy(note_texts)
    m["entropy_ratio"] = m["vocab_entropy"] / m["notes_vocab_entropy"] if m["notes_vocab_entropy"] else 0.0
    return m


def dedupe(E, threshold: float, seed: int = 0) -> list[int]:
    """Greedy near-duplicate filter on L2-normalised embeddings E (n x d). Returns kept indices."""
    import numpy as np
    order = list(range(len(E)))
    random.Random(seed).shuffle(order)
    kept: list[int] = []
    for i in order:
        if kept and float((E[kept] @ E[i]).max()) > threshold:
            continue
        kept.append(i)
    return sorted(kept)


def select_diverse(E, kinds: list[str], note_seqs: list[list[int]], target: int, quotas: dict[str, int],
                   min_per_fact: int, cap_per_fact: int, all_notes: list[int]) -> list[int]:
    """Greedy max-min (farthest-point) selection with per-fact coverage and per-kind quotas.

    E: L2-normalised embeddings. Phase 1 guarantees each fact up to `min_per_fact` dreams
    (round-robin over facts); phase 2 fills toward `target` respecting kind quotas and a
    per-fact cap; if quotas leave a shortfall they are relaxed.
    """
    import numpy as np
    n = len(E)
    chosen: list[int] = []
    in_set = np.zeros(n, bool)
    maxsim = np.full(n, -1.0)  # similarity to nearest chosen item (lower = more novel)
    count_fact = Counter()
    count_kind = Counter()
    by_fact: dict[int, list[int]] = {}
    for i, ns in enumerate(note_seqs):
        for s in ns:
            by_fact.setdefault(s, []).append(i)

    def take(i: int):
        chosen.append(i)
        in_set[i] = True
        count_kind[kinds[i]] += 1
        for s in note_seqs[i]:
            count_fact[s] += 1
        maxsim[:] = np.maximum(maxsim, E @ E[i])

    def best(cands, respect_quota=True):
        ok = [i for i in cands if not in_set[i]
              and (not respect_quota or count_kind[kinds[i]] < quotas.get(kinds[i], target))
              and all(count_fact[s] < cap_per_fact for s in note_seqs[i])]
        return min(ok, key=lambda i: maxsim[i]) if ok else None

    for r in range(min_per_fact):  # coverage, round-robin
        for f in all_notes:
            if len(chosen) >= target:
                break
            if count_fact[f] <= r:
                i = best(by_fact.get(f, []))
                if i is not None:
                    take(i)
    for respect in (True, False):
        while len(chosen) < target:
            i = best(range(n), respect)
            if i is None:
                break
            take(i)
    return sorted(chosen)


def collapse_check(baseline: dict | None, cur: dict, drop: float) -> list[str]:
    """Names of metrics that fell more than `drop` (relative) below night-1 baseline."""
    if not baseline:
        return []
    out = []
    for k in ("distinct_2", "embedding_dispersion"):
        if baseline[k] > 0 and cur[k] < (1 - drop) * baseline[k]:
            out.append(k)
    return out
