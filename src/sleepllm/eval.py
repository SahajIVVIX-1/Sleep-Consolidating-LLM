"""Evaluation: new-knowledge QA (without / with notebook) and the old-skill probe."""
from __future__ import annotations

import math
import random
import re
from pathlib import Path

import torch

from .data.synthetic_world import QA, answer_correct
from .llm import generate
from .notebook import Notebook
from .utils import read_jsonl, write_jsonl

PROBE_PATH = Path("data/old_skill_probe.jsonl")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _acc(rows: list[dict]) -> dict:
    n, k = len(rows), sum(r["correct"] for r in rows)
    lo, hi = wilson(k, n)
    return {"acc": k / n if n else 0.0, "n": n, "ci95": [lo, hi]}


# ---------------------------------------------------------------- new knowledge
def eval_new_knowledge(model, tok, qa: list[QA], notebook: Notebook | None = None, k_notes: int = 5,
                       max_new_tokens: int = 32) -> tuple[dict, list[dict]]:
    """Answers each question greedily. notebook=None -> no notes in the prompt."""
    prompts = []
    for q in qa:
        if notebook is None:
            prompts.append(q.question)
        else:
            ctx = notebook.format_prompt(q.question, k_notes)
            prompts.append(f"Notes:\n{ctx}\n\nQuestion: {q.question}\nAnswer briefly using the notes.")
    preds = generate(model, tok, prompts, max_new_tokens=max_new_tokens, batch_size=16)
    rows = [{"fact_id": q.fact_id, "relation": q.relation, "split": q.split, "is_update": q.is_update,
             "question": q.question, "gold": q.answer, "pred": p[0], "correct": answer_correct(p[0], q.answer)}
            for q, p in zip(qa, preds)]
    summ = {"all": _acc(rows),
            "heldout": _acc([r for r in rows if r["split"] == "heldout"]),
            "train": _acc([r for r in rows if r["split"] == "train"]),
            "updated_facts": _acc([r for r in rows if r["is_update"]]) if any(r["is_update"] for r in rows) else None}
    return summ, rows


# ---------------------------------------------------------------- old-skill probe
def build_probe(cfg, seed: int = 0, path: Path = PROBE_PATH) -> list[dict]:
    from datasets import load_dataset
    rng = random.Random(seed)
    items: list[dict] = []
    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    for i in rng.sample(range(len(arc)), cfg.probe_arc):
        r = arc[i]
        if r["answerKey"] not in r["choices"]["label"]:
            continue
        items.append({"task": "arc_easy", "ctx": f"Question: {r['question']}\nAnswer:",
                      "choices": [" " + c for c in r["choices"]["text"]],
                      "label": r["choices"]["label"].index(r["answerKey"])})
    hs = load_dataset("Rowan/hellaswag", split="validation")
    for i in rng.sample(range(len(hs)), cfg.probe_hellaswag):
        r = hs[i]
        items.append({"task": "hellaswag", "ctx": r["ctx"], "choices": [" " + e for e in r["endings"]],
                      "label": int(r["label"])})
    gs = load_dataset("openai/gsm8k", "main", split="test")
    for i in rng.sample(range(len(gs)), cfg.probe_gsm8k):
        r = gs[i]
        items.append({"task": "gsm8k", "question": r["question"],
                      "gold": r["answer"].split("####")[-1].strip().replace(",", "")})
    write_jsonl(path, items)
    return items


def load_probe(cfg, path: Path = PROBE_PATH) -> list[dict]:
    items = read_jsonl(path)
    return items if items else build_probe(cfg, path=path)


@torch.no_grad()
def _choice_scores(model, tok, ctx: str, choices: list[str]) -> list[float]:
    """Length-normalised log-likelihood of each continuation."""
    ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"]
    scores = []
    for ch in choices:
        ch_ids = tok(ctx + ch, add_special_tokens=False)["input_ids"][len(ctx_ids):]
        ids = torch.tensor([ctx_ids + ch_ids], device=model.device)
        logp = torch.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1)
        tgt = ids[0, 1:]
        lp = logp[torch.arange(len(tgt)), tgt][len(ctx_ids) - 1:]
        scores.append(lp.sum().item() / max(len(ch_ids), 1))
    return scores


def _last_number(t: str) -> str | None:
    nums = re.findall(r"-?\d[\d,]*\.?\d*", t)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def eval_old_skills(model, tok, probe: list[dict], gsm_tokens: int = 320) -> tuple[dict, list[dict]]:
    rows = []
    for it in probe:
        if it["task"] == "gsm8k":
            continue
        s = _choice_scores(model, tok, it["ctx"], it["choices"])
        rows.append({"task": it["task"], "correct": max(range(len(s)), key=s.__getitem__) == it["label"]})
    gs = [it for it in probe if it["task"] == "gsm8k"]
    if gs:
        outs = generate(model, tok, [f"{g['question']}\nSolve step by step. End with 'The answer is N.'" for g in gs],
                        max_new_tokens=gsm_tokens, batch_size=8)
        for g, o in zip(gs, outs):
            pred = _last_number(o[0])
            try:
                ok = pred is not None and float(pred) == float(g["gold"])
            except ValueError:
                ok = False
            rows.append({"task": "gsm8k", "correct": ok})
    summ = {t: _acc([r for r in rows if r["task"] == t]) for t in sorted({r["task"] for r in rows})}
    summ["overall"] = _acc(rows)
    return summ, rows


@torch.no_grad()
def wikitext_ppl(model, tok, n_tokens: int = 50000, window: int = 256) -> float:
    from datasets import load_dataset
    text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
    ids = tok(text, add_special_tokens=False)["input_ids"][:n_tokens]
    nll, cnt = 0.0, 0
    for s in range(0, len(ids) - window, window):
        x = torch.tensor([ids[s:s + window]], device=model.device)
        logits = model(input_ids=x).logits[0, :-1].float()
        nll += torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="sum").item()
        cnt += window - 1
    return math.exp(nll / cnt)
