"""Regulariser correctness: sign, which side of the threshold is kept, normalisation, quotas, replay v2."""
from collections import Counter

import numpy as np

from sleepllm.diversity import dedupe, distinct_n, matched_metrics, select_diverse
from sleepllm.replay import ReplayBuffer


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def vec_with_cos(c):
    """Unit vector with cosine c to e1."""
    return np.array([c, np.sqrt(1 - c * c)])


def test_dedupe_threshold_side():
    e1 = np.array([1.0, 0.0])
    above = np.vstack([e1, vec_with_cos(0.95)])   # cos 0.95 > 0.92 -> near duplicate -> one is dropped
    below = np.vstack([e1, vec_with_cos(0.90)])   # cos 0.90 <= 0.92 -> both kept
    assert len(dedupe(above, 0.92)) == 1
    assert len(dedupe(below, 0.92)) == 2


def test_dedupe_is_scale_invariant():
    """Un-normalised embeddings must give the same decisions as normalised ones (cosine, not dot product)."""
    rng = np.random.default_rng(0)
    E = rng.normal(size=(40, 6))
    scaled = E * rng.uniform(0.1, 20, size=(40, 1))
    assert dedupe(E, 0.7, seed=3) == dedupe(scaled, 0.7, seed=3)


def test_select_prefers_outlier_over_near_duplicate():
    # a, a' nearly identical, o is far away. Choosing 2 of 3 must give {o, one of the near-duplicates}.
    E = np.array([[1, 0], [0.999, 0.04], [0, 1]], float)
    sel = select_diverse(E, ["qa"] * 3, [[0], [0], [0]], target=2, quotas={"qa": 3}, min_per_fact=0,
                         cap_per_fact=5, all_notes=[0])
    assert 2 in sel and len(sel) == 2
    # and the result is the most dispersed pair
    pairs = {(0, 1): float(E[0] @ E[1]), (0, 2): 0.0, (1, 2): float(E[1] @ E[2])}
    assert tuple(sel) in {k for k, v in pairs.items() if v == min(pairs.values())}


def test_select_is_scale_invariant():
    rng = np.random.default_rng(2)
    E = rng.normal(size=(30, 5))
    kinds, facts = ["qa"] * 30, [[i % 3] for i in range(30)]
    a = select_diverse(E, kinds, facts, 9, {"qa": 9}, 0, 9, [0, 1, 2])
    b = select_diverse(E * rng.uniform(0.1, 9, size=(30, 1)), kinds, facts, 9, {"qa": 9}, 0, 9, [0, 1, 2])
    assert a == b


def test_selected_more_diverse_than_rejected():
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(6, 12))
    E = np.vstack([centers[i % 6] + 0.05 * rng.normal(size=12) for i in range(120)])
    sel = select_diverse(E, ["qa"] * 120, [[i % 6] for i in range(120)], 12, {"qa": 12}, 0, 12, [])
    rest = [i for i in range(120) if i not in sel]
    U = E / np.linalg.norm(E, axis=1, keepdims=True)

    def disp(ix):
        S = U[ix] @ U[ix].T
        return 1 - (S.sum() - len(ix)) / (len(ix) * (len(ix) - 1))
    assert disp(sel) > disp(rest[:12])


def test_relaxation_cap_does_not_flood_one_kind():
    """When the quota'd kinds are scarce, the fill must NOT be topped up with unlimited counterfactuals."""
    rng = np.random.default_rng(3)
    E = rng.normal(size=(100, 8))
    kinds = ["qa"] * 10 + ["counterfactual"] * 90
    facts = [[i % 10] for i in range(100)]
    sel = select_diverse(E, kinds, facts, target=40, quotas={"qa": 20, "counterfactual": 6}, min_per_fact=0,
                         cap_per_fact=50, all_notes=[], relax_factor=1.5)
    c = Counter(kinds[i] for i in sel)
    assert c["counterfactual"] <= 9          # ceil(1.5 * 6)
    assert len(sel) < 40                      # smaller set rather than a counterfactual-dominated one


def test_coverage_each_fact_gets_min_dreams():
    rng = np.random.default_rng(4)
    E = rng.normal(size=(60, 6))
    facts = [[i % 6] for i in range(60)]
    sel = select_diverse(E, ["qa"] * 60, facts, 24, {"qa": 24}, 3, 10, list(range(6)))
    c = Counter(f for i in sel for f in facts[i])
    assert all(c[f] >= 3 for f in range(6))


def test_matched_metrics_removes_size_effect():
    """distinct-2 of the full set falls with size; the size-matched value does not depend on how much data there is."""
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(60)]
    big = [" ".join(rng.choice(vocab, 8)) for _ in range(400)]
    small = big[:100]
    assert distinct_n(big, 2) < distinct_n(small, 2)
    mb, ms = matched_metrics(big, k=100), matched_metrics(small, k=100)
    assert abs(mb["distinct_2_m"] - ms["distinct_2_m"]) < 0.03


# ---- replay v2
def dreams(day, n, kind="qa", seq0=0):
    return [{"user": f"q{day}-{i}", "assistant": "a", "kind": kind, "note_seqs": [seq0 + i]} for i in range(n)]


def test_sample_mix_fractions_and_day_balance():
    old = [{"user": f"o{i}", "assistant": "a"} for i in range(100)]
    rb = ReplayBuffer(old, 1000, 0)
    for d in (1, 2, 3):
        rb.add_dreams(dreams(d, 100, seq0=d * 1000), d)
    out, st = rb.sample_mix(60, past_frac=0.4, old_frac=0.2)     # total = 60 / 0.4 = 150 -> 60 past + 30 old
    assert st["n_past"] == 60 and st["n_old"] == 30 and len(out) == 90
    assert max(st["past_by_day"].values()) - min(st["past_by_day"].values()) <= 1  # equal share per day


def test_sample_mix_excludes_counterfactuals_and_prunes_stale():
    rb = ReplayBuffer([{"user": "o", "assistant": "a"}] * 10, 1000, 0)
    rb.add_dreams(dreams(1, 20, "counterfactual"), 1)
    rb.add_dreams(dreams(1, 20, "qa", seq0=100), 1)
    out, st = rb.sample_mix(30, 0.4, 0.0)
    assert st["n_past"] == 20            # only the 20 real QA dreams are eligible
    dropped = rb.prune(active_seqs={q for q in range(100, 110)})   # facts 110-119 and all CF were superseded/inactive
    assert dropped == 20 + 10
    assert len(rb.reservoir) == 10


def test_sample_mix_first_night_has_no_past():
    rb = ReplayBuffer([{"user": "o", "assistant": "a"}] * 50, 1000, 0)
    out, st = rb.sample_mix(60, 0.4, 0.2)
    assert st["n_past"] == 0 and st["n_old"] == 30


def test_disable_fused_4bit_gemm_sets_fallback(monkeypatch):
    """Crash mitigation: bitsandbytes must be told to use dequant+linear (custom kernel max M = 0)."""
    import pytest
    ops = pytest.importorskip("bitsandbytes.backends.cuda.ops")
    from sleepllm.llm import disable_fused_4bit_gemm
    monkeypatch.setattr(ops, "_gemm_4bit_custom_max_m", 16, raising=False)
    monkeypatch.delenv("SLEEPLLM_FUSED_4BIT_GEMM", raising=False)
    assert disable_fused_4bit_gemm() is True and ops._gemm_4bit_custom_max_m == 0
    monkeypatch.setattr(ops, "_gemm_4bit_custom_max_m", 16, raising=False)
    monkeypatch.setenv("SLEEPLLM_FUSED_4BIT_GEMM", "1")
    assert disable_fused_4bit_gemm() is False and ops._gemm_4bit_custom_max_m == 16
