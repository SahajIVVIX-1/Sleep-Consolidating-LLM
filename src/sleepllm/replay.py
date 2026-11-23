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


def assert_disjoint(replay: list[dict], probe: list[dict]) -> None:
    """No replay prompt may equal a probe item (probe items come from other splits anyway)."""
    probe_txt = set()
    for p in probe:
        probe_txt.add(p.get("ctx") or p.get("question"))
    bad = [r for r in replay if any(t and t in r["user"] for t in probe_txt)]
    assert not bad, f"{len(bad)} replay items overlap the probe, e.g. {bad[0]['user'][:80]}"


class ReplayBuffer:
    """Fixed old-skill items + reservoir (Algorithm R) of past dreams."""

    def __init__(self, old_skill: list[dict], reservoir_cap: int = 300, seed: int = 0):
        self.old = list(old_skill)
        self.cap = reservoir_cap
        self.reservoir: list[dict] = []
        self.seen = 0
        self.rng = random.Random(seed)

    def add_dreams(self, samples: list[dict]) -> None:
        for s in samples:
            self.seen += 1
            item = {"user": s["user"], "assistant": s["assistant"], "kind": s.get("kind", "dream")}
            if len(self.reservoir) < self.cap:
                self.reservoir.append(item)
            else:
                j = self.rng.randrange(self.seen)
                if j < self.cap:
                    self.reservoir[j] = item

    def sample(self, n_new: int, ratio: float) -> list[dict]:
        """Replay items so that replay is `ratio` of the final mix: n_replay = n_new * ratio / (1 - ratio).
        Half from old skills, half from past dreams (all old-skill while the reservoir is empty)."""
        if ratio <= 0 or n_new <= 0:
            return []
        n_rep = round(n_new * ratio / (1 - ratio))
        n_dream = min(n_rep // 2, len(self.reservoir))
        n_old = min(n_rep - n_dream, len(self.old))
        out = self.rng.sample(self.old, n_old) + self.rng.sample(self.reservoir, n_dream)
        return [{"user": o["user"], "assistant": o["assistant"]} for o in out]
