import numpy as np
from sleepllm.diversity import distinct_n, self_bleu, embedding_dispersion, all_metrics


def fake_embed(xs):
    # bag-of-chars embedding, no model download
    out = []
    for x in xs:
        v = np.zeros(26)
        for ch in x.lower():
            if "a" <= ch <= "z":
                v[ord(ch) - 97] += 1
        out.append(v)
    return np.array(out)


SAME = ["the cat sat on the mat"] * 10
DIFF = ["the cat sat on the mat", "quantum fields oscillate wildly today", "my sister bakes sourdough bread",
        "rivers carve deep canyons slowly", "he plays violin every evening", "storms approach from the west"]


def test_distinct_n():
    assert distinct_n(["a b c d"], 1) == 1.0
    assert distinct_n(["a a a a"], 1) == 0.25
    assert distinct_n(SAME, 2) < 0.2 < distinct_n(DIFF, 2)
    assert distinct_n([], 1) == 0.0


def test_self_bleu():
    assert self_bleu(SAME) > 0.9
    assert self_bleu(DIFF) < 0.2
    assert self_bleu(["only one"]) == 0.0


def test_dispersion():
    assert embedding_dispersion(SAME, fake_embed) < 1e-9
    assert embedding_dispersion(DIFF, fake_embed) > 0.05


def test_all_metrics_keys():
    m = all_metrics(DIFF, fake_embed)
    assert set(m) == {"n", "distinct_1", "distinct_2", "distinct_3", "self_bleu", "embedding_dispersion"}
