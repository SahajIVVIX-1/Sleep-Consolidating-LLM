"""Step A: diagnose the Phase 1 dream-QA failures, then compare old vs new generator (yield, diversity, time)."""
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sleepllm import dream as D  # noqa: E402
from sleepllm.config import load_config  # noqa: E402
from sleepllm.diversity import night_metrics  # noqa: E402
from sleepllm.llm import generate, load_model  # noqa: E402
from sleepllm.notebook import Notebook  # noqa: E402
from sleepllm.utils import seed_everything, write_json, write_jsonl  # noqa: E402

OLD_QA_PROMPT = """Note: {note}

Write one question that can be answered using only this note, and its short answer.
Use exactly this format:
Q: <question>
A: <answer>"""

cfg = load_config("configs/dev_0p5b.yaml")
out = Path("results/phase2_stepA")
out.mkdir(parents=True, exist_ok=True)
seed_everything(0)
nb = Notebook("results/phase1/notebook.jsonl")
notes = nb.active()
tok, model = load_model(cfg.model.name)

# ---- 1. old generator (Phase 1 settings: temp 0.8, 3 samples/note, no retries)
t0 = time.time()
outs = generate(model, tok, [OLD_QA_PROMPT.format(note=n.text) for n in notes], max_new_tokens=64, temperature=0.8,
                top_p=0.95, n=3, batch_size=8)
old_secs = time.time() - t0
rows, seen = [], set()
for n, cands in zip(notes, outs):
    for c in cands:
        hit = len(tok(c)["input_ids"]) >= 63
        st, q, a = D.classify_qa(c, [n], seen, hit)
        if st == "ok":
            seen.add((D.normalize(q), D.normalize(a)))
        rows.append({"note": n.text, "raw": c, "status": st, "q": q, "a": a})
counts = Counter(r["status"] for r in rows)
rej = [r for r in rows if r["status"] != "ok"]
random.Random(0).shuffle(rej)
write_jsonl(out / "old_rejected_sample50.jsonl", rej[:50])
write_jsonl(out / "old_all_raw.jsonl", rows)
sample_counts = Counter(r["status"] for r in rej[:50])
old_valid = [r for r in rows if r["status"] == "ok"]
old_texts = [f"{r['q']} {r['a']}" for r in old_valid]
old = {"n_raw": len(rows), "counts_all": dict(counts), "counts_in_50_sample": dict(sample_counts),
       "yield": counts["ok"] / len(rows), "seconds": old_secs,
       "metrics_valid_qa": night_metrics(old_texts, [n.text for n in notes]) if len(old_texts) > 1 else None}
print("OLD", json.dumps(old, indent=1), flush=True)

# ---- 2. new generator, qa only (same raw budget: 150 slots) + full mix
seed_everything(0)
st = D.DreamState()
c2 = load_config("configs/dev_0p5b.yaml")
c2.dream.mix = {"qa": 1.0}
c2.dream.samples_per_note = 3
t0 = time.time()
samples, stats = D.generate_candidates(model, tok, notes, c2.dream, st, 1.0, 0)
new_secs = time.time() - t0
new_texts = [f"{s['user']} {s['assistant']}" for s in samples]
new = {"stats": stats, "seconds": new_secs, "metrics_valid_qa": night_metrics(new_texts, [n.text for n in notes])}
print("NEW", json.dumps({k: v for k, v in new.items()}, indent=1), flush=True)
write_jsonl(out / "new_qa_samples.jsonl", samples)
write_json(out / "stepA.json", {"old": old, "new": new})
