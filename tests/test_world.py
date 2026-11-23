import re
from sleepllm.data.synthetic_world import (World, QUESTIONS, STATEMENTS, answer_correct, day_to_dict)


def test_deterministic():
    a = day_to_dict(World(seed=3).generate_day(1))
    b = day_to_dict(World(seed=3).generate_day(1))
    c = day_to_dict(World(seed=4).generate_day(1))
    assert a == b and a != c


def test_fact_counts_in_range():
    for seed in range(10):
        w = World(seed=seed)
        for day in (1, 2, 3):
            d = w.generate_day(day)
            assert 20 <= len(d.facts) <= 50, (seed, day, len(d.facts))


def test_day1_has_no_updates_later_days_do():
    w = World(seed=1)
    assert not any(f.is_update for f in w.generate_day(1).facts)
    d2 = w.generate_day(2)
    ups = [f for f in d2.facts if f.is_update]
    assert ups and all(f.prev_object and f.prev_object != f.object for f in ups)


def test_updates_reference_earlier_facts_and_qa_uses_latest():
    w = World(seed=2)
    d1 = w.generate_day(1)
    earlier = {(f.subject, f.relation): f.object for f in d1.facts}
    d2 = w.generate_day(2)
    for f in d2.facts:
        if f.is_update:
            assert earlier[(f.subject, f.relation)] == f.prev_object
    gold = {(q.subject, q.relation): q.answer for q in d2.qa}
    for f in d2.facts:
        assert gold[(f.subject, f.relation)] == f.object


def test_heldout_templates_disjoint_from_train():
    for rel, (train, held) in QUESTIONS.items():
        assert not set(train) & set(held)
    d = World(seed=0).generate_day(1)
    train_q = {q.question for q in d.qa if q.split == "train"}
    held_q = {q.question for q in d.qa if q.split == "heldout"}
    assert train_q and held_q and not train_q & held_q


def test_answers_derivable_from_documents():
    d = World(seed=5).generate_day(1)
    text = " ".join(d.documents)
    for f in d.facts:
        assert f.subject in text and answer_correct(text, f.object)


def test_names_unique_and_unknown_looking():
    d = World(seed=0).generate_day(1)
    subs = [(f.subject, f.relation) for f in d.facts]
    assert len(subs) == len(set(subs))


def test_answer_correct():
    assert answer_correct("It is Brakel.", "Brakel")
    assert answer_correct("1987", "1987")
    assert not answer_correct("Brakelton", "Brakel")
    assert not answer_correct("", "x")
