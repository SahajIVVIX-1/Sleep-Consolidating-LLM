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


def _unit(E):
    import numpy as np
    E = np.asarray(E, dtype=float)
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)


def matched_metrics(texts: Sequence[str], E=None, k: int = 100, reps: int = 8, seed: int = 0) -> dict:
    """Size-matched diversity: mean over `reps` random subsets of size min(k, n).

    distinct-n and self-BLEU depend on set size, so sets of different sizes are only comparable at equal k.
    E: optional (n x d) embeddings for the dispersion; if None dispersion is skipped."""
    import numpy as np
    texts = list(texts)
    n = len(texts)
    kk = min(k, n)
    if kk < 2:
        return {"k": kk, "distinct_2_m": 0.0, "self_bleu_m": 0.0, "embedding_dispersion_m": 0.0}
    U = _unit(E) if E is not None else None
    rng = random.Random(seed)
    d2, sb, ed = [], [], []
    for _ in range(reps):
        idx = rng.sample(range(n), kk)
        sub = [texts[i] for i in idx]
        d2.append(distinct_n(sub, 2))
        sb.append(self_bleu(sub, max_samples=kk))
        if U is not None:
            S = U[idx] @ U[idx].T
            ed.append(1 - (S.sum() - np.trace(S)) / (kk * (kk - 1)))
    return {"k": kk, "distinct_2_m": float(np.mean(d2)), "self_bleu_m": float(np.mean(sb)),
            "embedding_dispersion_m": float(np.mean(ed)) if ed else 0.0}


def night_metrics(dream_texts: Sequence[str], note_texts: Sequence[str], embed=None) -> dict:
    import numpy as np
    texts = list(dream_texts)
    embed = embed or default_embedder()
    E = np.asarray(embed(texts), dtype=float) if len(texts) > 1 else None
    lookup = {t: E[i] for i, t in enumerate(texts)} if E is not None else None
    m = all_metrics(texts, (lambda xs: np.stack([lookup[x] for x in xs])) if lookup else embed)
    m.update(matched_metrics(texts, E))
    m["vocab_entropy"] = vocab_entropy(texts)
    m["notes_vocab_entropy"] = vocab_entropy(note_texts)
    m["entropy_ratio"] = m["vocab_entropy"] / m["notes_vocab_entropy"] if m["notes_vocab_entropy"] else 0.0
    return m


def dedupe(E, threshold: float, seed: int = 0) -> list[int]:
    """Greedy near-duplicate filter. Embeddings are L2-normalised here (cosine); an item is DROPPED when its
    cosine similarity to an already kept item is strictly greater than `threshold`. Returns kept indices."""
    U = _unit(E)
    order = list(range(len(U)))
    random.Random(seed).shuffle(order)
    kept: list[int] = []
    for i in order:
        if kept and float((U[kept] @ U[i]).max()) > threshold:
            continue
        kept.append(i)
    return sorted(kept)


def select_diverse(E, kinds: list[str], note_seqs: list[list[int]], target: int, quotas: dict[str, int],
                   min_per_fact: int, cap_per_fact: int, all_notes: list[int], relax_factor: float = 1.5) -> list[int]:
    """Greedy max-min (farthest-point) selection with per-fact coverage and per-kind quotas.

    At every step the candidate whose NEAREST already-chosen neighbour is FARTHEST away is taken (lowest
    max-similarity), so more diverse items are preferred. Phase 1 gives each fact up to `min_per_fact` dreams
    (round-robin over facts); phase 2 fills toward `target` within kind quotas and a per-fact cap; phase 3 relaxes
    quotas but never beyond relax_factor x quota per kind (a short pool yields a smaller set instead of filling it
    with one low-diversity kind). Embeddings are L2-normalised internally.
    """
    import numpy as np
    E = _unit(E)
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

    def best(cands, limit):
        ok = [i for i in cands if not in_set[i] and count_kind[kinds[i]] < limit(kinds[i])
              and all(count_fact[s] < cap_per_fact for s in note_seqs[i])]
        return min(ok, key=lambda i: maxsim[i]) if ok else None

    strict = lambda k: quotas.get(k, target)  # noqa: E731
    relaxed = lambda k: math.ceil(relax_factor * quotas.get(k, target))  # noqa: E731
    for r in range(min_per_fact):  # coverage, round-robin over facts
        for f in all_notes:
            if len(chosen) >= target:
                break
            if count_fact[f] <= r:
                i = best(by_fact.get(f, []), strict)
                if i is not None:
                    take(i)
    for limit in (strict, relaxed):
        while len(chosen) < target:
            i = best(range(n), limit)
            if i is None:
                break
            take(i)
    return sorted(chosen)


def collapse_check(baseline: dict | None, cur: dict, drop: float,
                   keys: Sequence[str] = ("distinct_2_m", "embedding_dispersion_m")) -> list[str]:
    """Names of metrics that fell more than `drop` (relative) below the night-1 baseline."""
    if not baseline:
        return []
    return [k for k in keys if k in baseline and k in cur and baseline[k] > 0 and cur[k] < (1 - drop) * baseline[k]]
