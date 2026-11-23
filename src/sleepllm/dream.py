"""Night, step 1: the model dreams practice data (QA, paraphrases, counterfactuals) from notes.

Naive version (Phase 1): plain temperature sampling, only light validity filters, no
diversity control. Held-out question templates are never used here.
"""
from __future__ import annotations

import re

from .data.synthetic_world import answer_correct, normalize
from .llm import generate
from .notebook import Note, Notebook

QA_PROMPT = """Note: {note}

Write one question that can be answered using only this note, and its short answer.
Use exactly this format:
Q: <question>
A: <answer>"""

PARA_PROMPT = """Note: {note}

Rewrite this note as a single sentence in different words. Keep every name, number and place exactly the same. Output only the sentence."""

CF_PROMPT = """Note: {note}

Write a hypothetical, made-up variation of this note in which one detail is different. Begin with "In an alternate world,". Output only the sentence."""

# how the samples-per-note budget is split across kinds
MIX = {"qa": 3, "paraphrase": 2, "counterfactual": 1}


def _split(n: int, kinds) -> dict[str, int]:
    tot = sum(MIX[k] for k in kinds)
    return {k: max(1, round(n * MIX[k] / tot)) for k in kinds}


def _parse_qa(t: str):
    m = re.search(r"Q:\s*(.+?)\s*\n\s*A:\s*(.+)", t, re.S)
    if not m:
        return None
    q, a = m.group(1).strip(), m.group(2).strip().split("\n")[0].strip()
    return (q, a) if q.endswith("?") and 0 < len(a.split()) <= 14 else None


def _valid_qa(q: str, a: str, note: Note) -> bool:
    na = normalize(a)
    return na != "" and na in normalize(note.text)


def dream(model, tok, notebook: Notebook, cfg, notes: list[Note] | None = None) -> tuple[list[dict], dict]:
    """Returns (samples, stats). Each sample: {kind, user, assistant, note_seq}."""
    notes = notes if notes is not None else notebook.active()
    split = _split(cfg.samples_per_note, cfg.kinds)
    samples: list[dict] = []
    stats = {"raw": {}, "kept": {}}
    for kind, n in split.items():
        prompt = {"qa": QA_PROMPT, "paraphrase": PARA_PROMPT, "counterfactual": CF_PROMPT}[kind]
        outs = generate(model, tok, [prompt.format(note=x.text) for x in notes],
                        max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature, top_p=cfg.top_p,
                        n=n, batch_size=max(1, 24 // n))
        raw = kept = 0
        for note, cands in zip(notes, outs):
            for c in cands:
                raw += 1
                if kind == "qa":
                    p = _parse_qa(c)
                    if not p or not _valid_qa(p[0], p[1], note):
                        continue
                    s = {"user": p[0], "assistant": p[1]}
                elif kind == "paraphrase":
                    c = c.strip().split("\n")[0]
                    if note.object and not (answer_correct(c, note.object) and normalize(note.subject) in normalize(c)):
                        continue
                    if not 4 <= len(c.split()) <= 60:
                        continue
                    topic = note.subject or "something you have learned"
                    s = {"user": f"Tell me a fact about {topic}.", "assistant": c}
                else:
                    c = c.strip().split("\n")[0]
                    if not c.lower().startswith("in an alternate world") or len(c.split()) > 60:
                        continue
                    s = {"user": f"Imagine an alternate world. Describe this differently: {note.text}", "assistant": c}
                kept += 1
                samples.append({"kind": kind, "note_seq": note.seq, **s})
        stats["raw"][kind], stats["kept"][kind] = raw, kept
    stats["n_samples"] = len(samples)
    stats["n_notes"] = len(notes)
    return samples, stats
