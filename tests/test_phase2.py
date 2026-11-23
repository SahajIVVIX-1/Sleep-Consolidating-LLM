import numpy as np
import pytest

from sleepllm.config import load_config
from sleepllm.diversity import collapse_check, dedupe, select_diverse, vocab_entropy
from sleepllm.dream import check_question, classify_qa, multihop_pairs, valid_text
from sleepllm.notebook import Note
from sleepllm.replay import ReplayBuffer, assert_disjoint


def note(seq, text, s=None, r=None, o=None):
    return Note(seq, 1, text, s, r, o)


# ---- dream validation
def test_check_question():
    ok = check_question("Question: Where does Mina Tor live?", "Mina Tor", "Dova", [], False)
    assert ok == ("ok", "Where does Mina Tor live?")
    assert check_question("Where does Mina Tor live in Dova?", "Mina Tor", "Dova", [], False)[0] == "off_topic"  # leak
    assert check_question("Where does she live?", "Mina Tor", "Dova", [], False)[0] == "off_topic"  # no subject
    assert check_question("Mina Tor lives where", "Mina Tor", "Dova", [], False)[0] == "format"
    assert check_question("Where does Mina Tor live?", "Mina Tor", "Dova", [], True)[0] == "truncated"
    assert check_question("Which city hosts the firm of Mina Tor?", "Mina Tor", "Dova", ["firm"], False)[0] == "off_topic"


def test_classify_old_qa():
    n = note(0, "Mina Tor lives in Dova.")
    assert classify_qa("Q: Where does Mina Tor live?\nA: Dova", [n], set(), False)[0] == "ok"
    assert classify_qa("Q: Where does Mina Tor live?\nA: Brakel", [n], set(), False)[0] == "ungrounded"
    assert classify_qa("What is the role?", [n], set(), False)[0] == "format"
    assert classify_qa("Q: Where does Mina Tor live?", [n], set(), False)[0] == "truncated"
    seen = {("where does mina tor live", "dova")}
    assert classify_qa("Q: Where does Mina Tor live?\nA: Dova", [n], seen, False)[0] == "duplicate"


def test_valid_text():
    n = note(0, "Mina Tor lives in Dova.", "Mina Tor", "home_city", "Dova")
    assert valid_text("paraphrase", "Dova is where Mina Tor lives.", n)
    assert valid_text("paraphrase", "Mina Tor lives in Dova.", n) is None  # copy
    assert valid_text("paraphrase", "Mina Tor lives in Brakel.", n) is None  # lost the object
    assert valid_text("counterfactual", "In an alternate world, Mina Tor lives in Brakel.", n)
    assert valid_text("counterfactual", "Mina Tor lives in Brakel.", n) is None


def test_multihop_pairs():
    a = note(0, "Pia works at Karst Labs.", "Pia", "employer", "Karst Labs")
    b = note(1, "Karst Labs is headquartered in Wenne.", "Karst Labs", "hq", "Wenne")
    c = note(2, "Zed lives in Wenne.", "Zed", "home_city", "Wenne")
    assert multihop_pairs([a, b, c]) == [(a, b)]


# ---- regulariser
def test_dedupe_drops_near_duplicates():
    E = np.array([[1, 0], [0.999, 0.04], [0, 1]], float)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    kept = dedupe(E, 0.95)
    assert len(kept) == 2 and 2 in kept


def test_select_diverse_coverage_and_quotas():
    rng = np.random.default_rng(0)
    n = 60
    E = rng.normal(size=(n, 8))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    kinds = ["qa"] * 30 + ["paraphrase"] * 30
    facts = [[i % 5] for i in range(n)]
    sel = select_diverse(E, kinds, facts, target=20, quotas={"qa": 10, "paraphrase": 10}, min_per_fact=3,
                         cap_per_fact=6, all_notes=list(range(5)))
    assert len(sel) == 20 and len(set(sel)) == 20
    from collections import Counter
    cf = Counter(f for i in sel for f in facts[i])
    assert all(cf[f] >= 3 for f in range(5))
    ck = Counter(kinds[i] for i in sel)
    assert ck["qa"] == 10 and ck["paraphrase"] == 10


def test_select_diverse_beats_random_on_dispersion():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(5, 16))
    E = np.vstack([base[i % 5] + 0.05 * rng.normal(size=16) for i in range(100)])  # 5 tight clusters
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    kinds, facts = ["qa"] * 100, [[i % 5] for i in range(100)]
    sel = select_diverse(E, kinds, facts, 10, {"qa": 10}, 0, 10, [])

    def disp(ix):
        S = E[ix] @ E[ix].T
        return 1 - (S.sum() - len(ix)) / (len(ix) * (len(ix) - 1))
    random_ix = list(range(0, 100, 10))
    assert disp(sel) > disp(random_ix)


def test_vocab_entropy_and_alarm():
    assert vocab_entropy(["a a a a"]) == 0.0
    assert vocab_entropy(["a b c d"]) == pytest.approx(2.0)
    base = {"distinct_2": 0.6, "embedding_dispersion": 0.7}
    assert collapse_check(base, {"distinct_2": 0.58, "embedding_dispersion": 0.69}, 0.15) == []
    assert collapse_check(base, {"distinct_2": 0.4, "embedding_dispersion": 0.69}, 0.15) == ["distinct_2"]
    assert collapse_check(None, base, 0.15) == []


# ---- replay
def test_replay_ratio_and_reservoir():
    old = [{"user": f"q{i}", "assistant": "a"} for i in range(100)]
    rb = ReplayBuffer(old, reservoir_cap=10, seed=0)
    s = rb.sample(70, 0.3)
    assert len(s) == 30  # 30% of the final mix of 100
    rb.add_dreams([{"user": f"d{i}", "assistant": "x", "kind": "qa"} for i in range(50)])
    assert len(rb.reservoir) == 10 and rb.seen == 50
    s = rb.sample(70, 0.3)
    assert len(s) == 30 and sum(1 for x in s if x["user"].startswith("d")) == 10
    assert rb.sample(70, 0.0) == []


def test_assert_disjoint():
    probe = [{"task": "arc_easy", "ctx": "Question: Which is hot?\nAnswer:"}, {"task": "gsm8k", "question": "2+2?"}]
    assert_disjoint([{"user": "something else", "assistant": "x"}], probe)
    with pytest.raises(AssertionError):
        assert_disjoint([{"user": "Question: Which is hot?\nAnswer:", "assistant": "x"}], probe)


def test_config_dict_override():
    c = load_config(None, ["dream.regulariser=true", "replay.ratio=0.5"])
    assert c.dream.regulariser is True and c.replay.ratio == 0.5 and c.dream.mix["qa"] == 0.45
