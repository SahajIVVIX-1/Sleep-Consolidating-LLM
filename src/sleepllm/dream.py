"""Night, step 1: the model dreams practice data from notes.

Kinds: qa, paraphrase, counterfactual, multihop. Every candidate is validated
(format / truncation / grounding / topic / duplicate) and rejected slots are retried.
Held-out question templates are never used here.

The regulariser (over-generate, dedupe, diverse selection, collapse alarm) lives in
`diversity.py`; `dream()` only produces validated candidates, `select_dreams()` applies it.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field

from .data.synthetic_world import answer_correct, normalize
from .llm import generate
from .notebook import Note

STOP = set("the a an of in on at to is are was were by for and or as with from its it this that who what which where when "
           "does do did has have had be been his her their into about than then".split())

# ---------------------------------------------------------------- prompts
QA_EXAMPLES = [  # (note, question, answer): invented, not world facts
    ("Orla Pennick lives in Sandrel.", "In which city does Orla Pennick live?", "Sandrel"),
    ("Dremko Works makes a product called Valzi-9.", "Which product is made by Dremko Works?", "Valzi-9"),
    ("Tavi Rond was born in 1966.", "What year was Tavi Rond born?", "1966"),
    ("Karst Labs is led by its chief executive, Ibo Mell.", "Who is the chief executive of Karst Labs?", "Ibo Mell"),
    ("Pia Norr works as a rope walker.", "What does Pia Norr do for work?", "rope walker"),
    ("Zaltik Corp has its main office in Wenne.", "Which company has its main office in Wenne?", "Zaltik Corp"),
]
QA_HEAD = ("Write one question that can be answered using only the note, and its short answer (at most 8 words, "
           "copied from the note).\nUse exactly this format and nothing else:\nQ: <question>\nA: <answer>\n\n")
MH_HEAD = ("Write one question that needs BOTH notes to answer, and its short final answer (at most 8 words, "
           "copied from one of the notes).\nUse exactly this format and nothing else:\nQ: <question>\nA: <answer>\n\n")
MH_EXAMPLE = ("Notes:\n- Pia Norr works at Karst Labs.\n- Karst Labs is headquartered in Wenne.\n"
              "Q: In which city is the company that employs Pia Norr headquartered?\nA: Wenne\n\n")

PARA_PROMPT = ("Note: {note}\n\nRewrite this note as a single sentence in different words. Keep every name, number and "
               "place exactly the same. Output only the sentence.")
CF_PROMPT = ("Note: {note}\n\nWrite a hypothetical, made-up variation of this note in which one detail is different. "
             'Begin with "In an alternate world,". Output only the sentence.')


AF_HEAD = ("Write one question about the subject of the note whose answer is exactly the given answer. "
           "The question must mention the subject by name and must not contain the answer. Output only the question.\n\n")
MH_AF_HEAD = ("Write one question that needs BOTH notes to answer and whose answer is exactly the given answer. "
              "The question must mention the first subject by name, must not mention the intermediate entity, and must "
              "not contain the answer. Output only the question.\n\n")
MH_AF_EXAMPLE = ("Notes:\n- Pia Norr works at Karst Labs.\n- Karst Labs is headquartered in Wenne.\nAnswer: Wenne\n"
                 "Question: In which city is the company that employs Pia Norr headquartered?\n\n")


def af_prompt(note_text: str, answer: str, variant: int) -> str:
    ex = [QA_EXAMPLES[variant % len(QA_EXAMPLES)], QA_EXAMPLES[(variant + 1) % len(QA_EXAMPLES)]]
    ex = [e for e in ex if not e[1].startswith("Which company has")]  # reverse questions are ambiguous
    shots = "".join(f"Note: {n}\nAnswer: {a}\nQuestion: {q}\n\n" for n, q, a in ex)
    return AF_HEAD + shots + f"Note: {note_text}\nAnswer: {answer}\nQuestion:"


def mh_af_prompt(t1: str, t2: str, answer: str) -> str:
    return MH_AF_HEAD + MH_AF_EXAMPLE + f"Notes:\n- {t1}\n- {t2}\nAnswer: {answer}\nQuestion:"


def check_question(raw: str, subject: str, answer: str, forbid: list[str], hit_limit: bool) -> tuple[str, str | None]:
    """Validate an answer-first question. Returns (status, question)."""
    q = re.sub(r"^\s*(Question:)?\s*", "", raw.strip().split("\n")[0]).strip()
    if hit_limit:
        return "truncated", None
    if not q.endswith("?") or not 3 <= len(q.split()) <= 30:
        return "format", None
    nq = normalize(q)
    if normalize(answer) in nq or any(normalize(f) in nq for f in forbid if f):
        return "off_topic", None  # leaks the answer / intermediate entity
    if normalize(subject) not in nq:
        return "off_topic", None  # does not pin down the subject -> ambiguous
    return "ok", q


def qa_prompt(note_text: str, variant: int) -> str:
    ex = [QA_EXAMPLES[variant % len(QA_EXAMPLES)], QA_EXAMPLES[(variant + 1) % len(QA_EXAMPLES)]]
    shots = "".join(f"Note: {n}\nQ: {q}\nA: {a}\n\n" for n, q, a in ex)
    return QA_HEAD + shots + f"Note: {note_text}"


def mh_prompt(t1: str, t2: str) -> str:
    return MH_HEAD + MH_EXAMPLE + f"Notes:\n- {t1}\n- {t2}"


# ---------------------------------------------------------------- validation
def _content(s: str) -> set[str]:
    return {w for w in normalize(s).split() if w not in STOP and len(w) > 2}


def parse_qa(t: str):
    """Returns ('ok', q, a) | ('truncated', ...) | ('format', ...)."""
    if "Q:" in t and "A:" not in t:
        return "truncated", None, None
    m = re.search(r"Q:\s*(.+?)\s*\n\s*A:\s*(.+)", t, re.S)
    if not m:
        return "format", None, None
    q, a = m.group(1).strip(), m.group(2).strip().split("\n")[0].strip()
    if not q.endswith("?") or not a or len(a.split()) > 14:
        return "format", None, None
    return "ok", q, a


def classify_qa(raw: str, notes: list[Note], seen: set, hit_limit: bool) -> tuple[str, str | None, str | None]:
    """Primary failure type for one raw QA output; 'ok' if valid."""
    status, q, a = parse_qa(raw)
    if status != "ok":
        return status, None, None
    if hit_limit:
        return "truncated", None, None
    ntext = " ".join(n.text for n in notes)
    na = normalize(a)
    if not _content(q) & _content(ntext) or na in normalize(q):
        return "off_topic", None, None
    if not na or na not in normalize(ntext):
        return "ungrounded", None, None
    key = (normalize(q), na)
    if key in seen:
        return "duplicate", None, None
    return "ok", q, a


def valid_text(kind: str, raw: str, note: Note) -> str | None:
    """Validation for paraphrase / counterfactual outputs. Returns cleaned text or None."""
    c = raw.strip().split("\n")[0].strip()
    if not 4 <= len(c.split()) <= 60 or not c.endswith((".", "!", '"')):
        return None
    if kind == "paraphrase":
        if normalize(c) == normalize(note.text):
            return None
        if note.object and not (answer_correct(c, note.object) and normalize(note.subject) in normalize(c)):
            return None
    elif kind == "counterfactual":
        if not c.lower().startswith("in an alternate world"):
            return None
    return c


# ---------------------------------------------------------------- generation
@dataclass
class DreamState:
    """Mutable across nights: the collapse alarm raises these."""
    temp_bump: float = 0.0
    variety_bump: int = 0
    baseline: dict | None = None
    alarms: list = field(default_factory=list)


def _targets(cfg, n_notes: int, mult: float) -> dict[str, int]:
    total = cfg.samples_per_note * n_notes
    return {k: max(1, round(total * r * mult)) for k, r in cfg.mix.items() if r > 0}


def multihop_pairs(notes: list[Note]) -> list[tuple[Note, Note]]:
    by_subj: dict[str, list[Note]] = {}
    for n in notes:
        if n.subject:
            by_subj.setdefault(n.subject.lower(), []).append(n)
    return [(a, b) for a in notes if a.object and a.subject for b in by_subj.get(a.object.lower(), []) if b is not a]


def _temp(cfg, state: DreamState, kind: str) -> float:
    base = {"qa": cfg.qa_temp, "multihop": cfg.qa_temp, "paraphrase": cfg.para_temp, "counterfactual": cfg.cf_temp}[kind]
    return min(base + state.temp_bump, cfg.temp_cap)


def generate_candidates(model, tok, notes: list[Note], cfg, state: DreamState, mult: float = 1.0,
                        seed: int = 0) -> tuple[list[dict], dict]:
    """Produce validated dream candidates; mult>1 over-generates (regulariser)."""
    rng = random.Random(seed)
    targets = _targets(cfg, len(notes), mult)
    pairs = multihop_pairs(notes)
    keyed = [n for n in notes if n.subject and n.object]
    if not pairs:
        targets.pop("multihop", None)
    if not keyed:
        targets.pop("qa", None)
    n_var = min(cfg.variety_base + state.variety_bump, len(QA_EXAMPLES))
    samples: list[dict] = []
    stats = {"slots": {}, "first_pass_valid": {}, "final_valid": {}, "rejected": {}, "targets": targets,
             "attempts": {}}
    for kind, T in targets.items():
        # slot i -> (notes, variant); round-robin keeps coverage even
        slots = []
        for i in range(T):
            if kind == "multihop":
                a, b = pairs[i % len(pairs)]
                slots.append(([a, b], i % n_var))
            elif kind == "qa":
                slots.append(([keyed[i % len(keyed)]], i % n_var))
            else:
                slots.append(([notes[i % len(notes)]], i % n_var))
        rej = Counter()
        seen: set = set()
        pending = list(range(len(slots)))
        first_valid = 0
        attempts = 0
        for attempt in range(cfg.retries + 1):
            if not pending:
                break
            prompts = []
            for i in pending:
                ns, v = slots[i]
                v = v + attempt  # shift the few-shot pair on retry
                if kind == "qa":
                    prompts.append(af_prompt(ns[0].text, ns[0].object, v))
                elif kind == "multihop":
                    prompts.append(mh_af_prompt(ns[0].text, ns[1].text, ns[1].object))
                elif kind == "paraphrase":
                    prompts.append(PARA_PROMPT.format(note=ns[0].text))
                else:
                    prompts.append(CF_PROMPT.format(note=ns[0].text))
            outs = generate(model, tok, prompts, max_new_tokens=cfg.max_new_tokens, temperature=_temp(cfg, state, kind),
                            top_p=cfg.top_p, n=1, batch_size=24)
            attempts += len(pending)
            still = []
            for i, o in zip(pending, outs):
                raw = o[0]
                ns = slots[i][0]
                hit = len(tok(raw)["input_ids"]) >= cfg.max_new_tokens - 1
                if kind in ("qa", "multihop"):
                    a = ns[-1].object
                    st, q = check_question(raw, ns[0].subject, a, [ns[0].object] if kind == "multihop" else [], hit)
                    if st == "ok" and (normalize(q), normalize(a)) in seen:
                        st = "duplicate"
                    if st == "ok":
                        seen.add((normalize(q), normalize(a)))
                        s = {"user": q, "assistant": a}
                else:
                    c = None if hit else valid_text(kind, raw, ns[0])
                    st = "ok" if c else "format"
                    if c:
                        topic = ns[0].subject or "something you have learned"
                        s = ({"user": f"Tell me a fact about {topic}.", "assistant": c} if kind == "paraphrase" else
                             {"user": f"Imagine an alternate world. Describe this differently: {ns[0].text}",
                              "assistant": c})
                if st == "ok":
                    if attempt == 0:
                        first_valid += 1
                    samples.append({"kind": kind, "note_seqs": [n.seq for n in ns], **s})
                else:
                    rej[st] += 1
                    still.append(i)
            pending = still
        stats["slots"][kind] = len(slots)
        stats["attempts"][kind] = attempts
        stats["first_pass_valid"][kind] = first_valid
        stats["final_valid"][kind] = len(slots) - len(pending)
        stats["rejected"][kind] = dict(rej)
    stats["n_valid"] = len(samples)
    stats["first_pass_yield"] = sum(stats["first_pass_valid"].values()) / max(sum(stats["slots"].values()), 1)
    stats["final_fill"] = sum(stats["final_valid"].values()) / max(sum(stats["slots"].values()), 1)
    stats["yield_per_attempt"] = len(samples) / max(sum(stats["attempts"].values()), 1)
    return samples, stats


# ---------------------------------------------------------------- one night's dream set
def make_dreams(model, tok, notes: list[Note], cfg, state: DreamState, night: int, seed: int = 0,
                embed=None) -> tuple[list[dict], dict]:
    """Generate (and, if cfg.regulariser, over-generate + dedupe + diversity-select) tonight's dreams.

    Also computes night metrics and runs the collapse alarm. Returns (samples, stats)."""
    import time

    import numpy as np

    from .diversity import collapse_check, dedupe, default_embedder, night_metrics, select_diverse
    embed = embed or default_embedder()
    t0 = time.time()
    mult = cfg.overgen if cfg.regulariser else 1.0
    cands, stats = generate_candidates(model, tok, notes, cfg, state, mult, seed)
    stats["gen_seconds"] = time.time() - t0
    stats["temp_bump"], stats["variety_bump"] = state.temp_bump, state.variety_bump
    stats["n_candidates"] = len(cands)
    texts = [f"{c['user']} {c['assistant']}" for c in cands]
    if cfg.regulariser and cands:
        E = np.asarray(embed(texts), dtype=float)
        E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        keep = dedupe(E, cfg.dedupe_cos, seed)
        stats["n_after_dedupe"] = len(keep)
        Ek = E[keep]
        total = cfg.samples_per_note * len(notes)
        quotas = {k: max(1, round(total * r)) for k, r in cfg.mix.items() if r > 0}
        sel = select_diverse(Ek, [cands[i]["kind"] for i in keep], [cands[i]["note_seqs"] for i in keep], total,
                             quotas, cfg.min_per_fact, cfg.cap_per_fact, [n.seq for n in notes])
        samples = [cands[keep[i]] for i in sel]
    else:
        samples = cands
    stats["n_samples"] = len(samples)
    stats["by_kind"] = dict(Counter(s["kind"] for s in samples))
    stats["facts_covered"] = len({q for s in samples for q in s["note_seqs"]})
    stats["n_notes"] = len(notes)
    dtexts = [f"{s['user']} {s['assistant']}" for s in samples]
    stats["metrics"] = night_metrics(dtexts, [n.text for n in notes], embed)
    stats["seconds"] = time.time() - t0
    # collapse alarm: compare with night 1; only the regulariser acts on it
    if state.baseline is None:
        state.baseline = dict(stats["metrics"])
    stats["alarm"] = collapse_check(state.baseline, stats["metrics"], cfg.alarm_drop) if night > 1 else []
    if stats["alarm"] and cfg.regulariser:
        state.temp_bump = min(state.temp_bump + cfg.temp_bump, cfg.temp_cap - cfg.para_temp)
        state.variety_bump = min(state.variety_bump + 1, len(QA_EXAMPLES) - cfg.variety_base)
        state.alarms.append({"night": night, "metrics": stats["alarm"], "temp_bump": state.temp_bump,
                             "variety_bump": state.variety_bump})
        stats["alarm_action"] = {"temp_bump": state.temp_bump, "variety_bump": state.variety_bump}
    return samples, stats
