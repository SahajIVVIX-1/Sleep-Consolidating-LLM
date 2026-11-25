"""Replay: a fixed old-skill buffer (train splits only) plus a reservoir of dreams from past nights.

Old-skill data never contains probe items (probe = ARC-Easy test, HellaSwag validation, GSM8K test; replay uses
the TRAIN splits; `assert_disjoint` double-checks by text). Samples are {"user", "assistant"} chat pairs.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

from .utils import read_jsonl, write_jsonl

OLD_SKILL_PATH = Path("data/old_skill_replay.jsonl")


def _clean_gsm(ans: str) -> tuple[str, str]:
    body, final = ans.split("####")
    body = re.sub(r"<<.*?>>", "", body).strip()
    final = final.strip().replace(",", "")
    return f"{body}\nThe answer is {final}.", final


def build_old_skill(n: int = 400, seed: int = 0, path: Path = OLD_SKILL_PATH) -> list[dict]:
    """n items split evenly over ARC-Easy / HellaSwag / GSM8K (train splits) and alpaca-cleaned (short)."""
    from datasets import load_dataset
    rng = random.Random(seed)
    k = n // 4
    items: list[dict] = []
    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    for i in rng.sample(range(len(arc)), k * 2):
        r = arc[i]
        if r["answerKey"] in r["choices"]["label"] and len([x for x in items if x["task"] == "arc_easy"]) < k:
            ans = r["choices"]["text"][r["choices"]["label"].index(r["answerKey"])]
            items.append({"task": "arc_easy", "user": f"Question: {r['question']}\nAnswer:", "assistant": ans})
    hs = load_dataset("Rowan/hellaswag", split="train")
    for i in rng.sample(range(len(hs)), k):
        r = hs[i]
        items.append({"task": "hellaswag", "user": f"Complete the sentence: {r['ctx']}",
                      "assistant": r["endings"][int(r["label"])]})
    gs = load_dataset("openai/gsm8k", "main", split="train")
    got = 0
    for i in rng.sample(range(len(gs)), len(gs)):
        body, final = _clean_gsm(gs[i]["answer"])
        if len(body.split()) <= 90 and got < k:
            items.append({"task": "gsm8k", "user": f"{gs[i]['question']}\nSolve step by step. End with 'The answer is N.'",
                          "assistant": body})
            got += 1
    al = load_dataset("yahma/alpaca-cleaned", split="train")
    got, want = 0, n - len(items)
    for i in rng.sample(range(len(al)), 5000):
        r = al[i]
        if len(r["output"].split()) <= 60 and len(r["input"].split()) <= 60 and got < want:
            u = r["instruction"] + (f"\n\n{r['input']}" if r["input"] else "")
            items.append({"task": "alpaca", "user": u, "assistant": r["output"]})
            got += 1
    write_jsonl(path, items)
    return items


def load_old_skill(n: int = 400, seed: int = 0, path: Path = OLD_SKILL_PATH) -> list[dict]:
    items = read_jsonl(path)
    return items if len(items) >= n else build_old_skill(n, seed, path)


TEXT_PATH = Path("data/wikitext_replay.jsonl")


def split_paragraph(par: str, n_prompt: int = 25, n_target: int = 45) -> dict | None:
    """General-text replay item: continue a Wikipedia paragraph. None if the paragraph is too short."""
    w = par.split()
    if len(w) < n_prompt + n_target:
        return None
    return {"task": "wikitext", "user": "Continue the text:" + chr(10) + " ".join(w[:n_prompt]),
            "assistant": " ".join(w[n_prompt:n_prompt + n_target])}


def build_text_replay(n: int = 300, seed: int = 0, path: Path = TEXT_PATH) -> list[dict]:
    """n general-text items from the WikiText-2 TRAIN split (the perplexity probe uses the TEST split)."""
    from datasets import load_dataset
    rng = random.Random(seed)
    train = [t.strip() for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"]]
    pars = [t for t in train if t and not t.startswith("=") and len(t.split()) >= 70]
    rng.shuffle(pars)
    items = [it for it in (split_paragraph(p) for p in pars[: 3 * n]) if it][:n]
    write_jsonl(path, items)
    return items


def load_text_replay(n: int = 300, seed: int = 0, path: Path = TEXT_PATH) -> list[dict]:
    items = read_jsonl(path)
    return items if len(items) >= n else build_text_replay(n, seed, path)


def assert_text_disjoint_from_test(items: list[dict], test_paragraphs: list[str]) -> None:
    """No replayed text may come from the WikiText test split used for the perplexity probe."""
    test = " ".join(test_paragraphs)
    bad = [it for it in items if it["assistant"][:60] in test or it["user"].split(chr(10), 1)[-1][:60] in test]
    assert not bad, f"{len(bad)} text-replay items overlap the WikiText test split"


def assert_disjoint(replay: list[dict], probe: list[dict]) -> None:
    """No replay prompt may equal a probe item (probe items come from other splits anyway)."""
    probe_txt = set()
    for p in probe:
        probe_txt.add(p.get("ctx") or p.get("question"))
    bad = [r for r in replay if any(t and t in r["user"] for t in probe_txt)]
    assert not bad, f"{len(bad)} replay items overlap the probe, e.g. {bad[0]['user'][:80]}"


class ReplayBuffer:
    """Fixed old-skill items + reservoir of past dreams.

    v1 (`sample`): ratio of the mix, half old-skill / half uniformly drawn past dreams (any kind).
    v2 (`sample_mix`): fractions of the final batch; past dreams are day-balanced (every previous day gets an
    equal share), exclude counterfactuals (they are not true facts) and are pruned when a later note supersedes
    the fact they were generated from.
    """

    def __init__(self, old_skill: list[dict], reservoir_cap: int = 1500, seed: int = 0, text: list[dict] | None = None):
        self.old = list(old_skill)
        self.text = list(text or [])
        self.cap = reservoir_cap
        self.reservoir: list[dict] = []
        self.seen = 0
        self.rng = random.Random(seed)

    def add_dreams(self, samples: list[dict], day: int | None = None) -> None:
        for s in samples:
            self.seen += 1
            item = {"user": s["user"], "assistant": s["assistant"], "kind": s.get("kind", "dream"),
                    "day": day, "note_seqs": s.get("note_seqs", [])}
            if len(self.reservoir) < self.cap:
                self.reservoir.append(item)
            else:
                j = self.rng.randrange(self.seen)
                if j < self.cap:
                    self.reservoir[j] = item

    def prune(self, active_seqs: set[int]) -> int:
        """Drop dreams whose source note is no longer active (the fact was updated later). Returns #dropped."""
        before = len(self.reservoir)
        self.reservoir = [r for r in self.reservoir if all(q in active_seqs for q in r.get("note_seqs", []))]
        return before - len(self.reservoir)

    def sample(self, n_new: int, ratio: float) -> list[dict]:
        """v1: replay is `ratio` of the final mix. Half old skills, half past dreams (all old while empty)."""
        if ratio <= 0 or n_new <= 0:
            return []
        n_rep = round(n_new * ratio / (1 - ratio))
        n_dream = min(n_rep // 2, len(self.reservoir))
        n_old = min(n_rep - n_dream, len(self.old))
        out = self.rng.sample(self.old, n_old) + self.rng.sample(self.reservoir, n_dream)
        return [{"user": o["user"], "assistant": o["assistant"]} for o in out]

    def sample_mix(self, n_new: int, past_frac: float, old_frac: float, text_frac: float = 0.0) -> tuple[list[dict], dict]:
        """v2: batch = new + past + old (+ general text) with the given fractions of the FINAL batch."""
        if n_new <= 0 or past_frac + old_frac + text_frac >= 1:
            return [], {"n_past": 0, "n_old": 0, "n_text": 0}
        total = n_new / (1 - past_frac - old_frac - text_frac)
        n_past_want, n_old = round(total * past_frac), min(round(total * old_frac), len(self.old))
        n_text = min(round(total * text_frac), len(self.text))
        by_day: dict = {}
        for r in self.reservoir:
            if r["kind"] != "counterfactual":
                by_day.setdefault(r["day"], []).append(r)
        for v in by_day.values():
            self.rng.shuffle(v)
        past: list[dict] = []
        per_day: dict = {}
        days = sorted(by_day, key=lambda d: (d is None, d))
        while len(past) < n_past_want and any(by_day.values()):
            for d in days:  # round-robin over days -> equal weight per previous day
                if by_day[d] and len(past) < n_past_want:
                    past.append(by_day[d].pop())
                    per_day[d] = per_day.get(d, 0) + 1
        old = self.rng.sample(self.old, n_old)
        text = self.rng.sample(self.text, n_text)
        out = [{"user": o["user"], "assistant": o["assistant"]} for o in past + old + text]
        return out, {"n_past": len(past), "n_old": len(old), "n_text": len(text), "past_by_day": per_day}
