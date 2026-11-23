"""Day: the agent reads documents, extracts facts, writes notes to the notebook.

The note text is the source sentence; the model's job is to extract a structured
(subject | relation | object) key so that updates can supersede earlier notes.
Extraction quality is measured against the world's ground truth.
"""
from __future__ import annotations

import re

from .data.synthetic_world import RELATIONS, Fact, normalize
from .llm import generate
from .notebook import Notebook

PROMPT = """Extract one fact from the sentence as: subject | relation | object
Relations (person facts): employer (company a person works for), home_city (city a person lives in), role (a person's job), birth_year (year a person was born).
Relations (company facts): hq (city of headquarters), founded (year founded), ceo (person who leads it), product (what it makes).

Sentence: Lira Venn resides in Orsk.
Fact: Lira Venn | home_city | Orsk

Sentence: Tolmax Works is led by its chief executive, Dar Quell.
Fact: Tolmax Works | ceo | Dar Quell

Sentence: Pell Dromi is on the staff of Karvo Labs.
Fact: Pell Dromi | employer | Karvo Labs

Sentence: 1974 is the year Nev Sarok was born.
Fact: Nev Sarok | birth_year | 1974

Sentence: By trade, Mol Tessin is a stonemason.
Fact: Mol Tessin | role | stonemason

Sentence: Brivo Corp began operating in 1931.
Fact: Brivo Corp | founded | 1931

Sentence: The headquarters of Zenk Holdings are in Praal.
Fact: Zenk Holdings | hq | Praal

Sentence: Quor Dynamics is best known for Velka-12.
Fact: Quor Dynamics | product | Velka-12

Sentence: {sentence}
Fact:"""

_UPD = re.compile(r"^Update, this has changed:\s*")


def split_sentences(doc: str) -> list[str]:
    # sentences end with ". " followed by a capital; names contain no periods
    parts = re.split(r"(?<=\.)\s+(?=[A-Z0-9])", doc.strip())
    out = []
    for p in parts:
        out.append(p)
    return out


def parse_fact(line: str, sentence: str):
    line = re.sub(r"^\s*Fact:\s*", "", line.strip().split("\n")[0])
    parts = [p.strip() for p in line.split("|")]
    if len(parts) != 3 or parts[1] not in RELATIONS:
        return None
    s, r, o = parts
    if normalize(s) not in normalize(sentence) or normalize(o) not in normalize(sentence):
        return None
    return s, r, o


def run_day(model, tok, notebook: Notebook, day: int, documents: list[str], truth: list[Fact]) -> dict:
    sentences = [_UPD.sub("", s) for d in documents for s in split_sentences(d)]
    outs = generate(model, tok, [PROMPT.format(sentence=s) for s in sentences], max_new_tokens=32, batch_size=16)
    parsed = 0
    got = set()
    for s, o in zip(sentences, outs):
        p = parse_fact(o[0], s)
        if p:
            parsed += 1
            got.add((normalize(p[0]), p[1], normalize(p[2])))
            notebook.append(day, s, *p)
        else:
            notebook.append(day, s)
    tset = {(normalize(f.subject), f.relation, normalize(f.object)) for f in truth}
    return {
        "n_sentences": len(sentences), "n_truth_facts": len(tset), "n_parsed": parsed,
        "recall": len(tset & got) / len(tset), "precision": len(tset & got) / max(len(got), 1),
        "raw_examples": [{"sentence": s, "output": o[0]} for s, o in list(zip(sentences, outs))[:8]],
    }
