"""Step A2: size-matched diversity comparison old vs new QA generator, and QA temperature sweep."""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sleepllm import dream as D  # noqa: E402
from sleepllm.config import load_config  # noqa: E402
from sleepllm.diversity import distinct_n, embedding_dispersion, self_bleu  # noqa: E402
from sleepllm.llm import load_model  # noqa: E402
from sleepllm.notebook import Notebook  # noqa: E402
from sleepllm.utils import seed_everything, write_json  # noqa: E402


def matched(texts, k, reps=20):
    ms = []
    for r in range(reps):
        sub = random.Random(r).sample(texts, k)
        ms.append((distinct_n(sub, 1), distinct_n(sub, 2), self_bleu(sub), embedding_dispersion(sub)))
    return {n: sum(m[i] for m in ms) / reps for i, n in enumerate(["distinct_1", "distinct_2", "self_bleu", "embedding_dispersion"])}


out = Path("results/phase2_stepA")
old = [json.loads(l) for l in open(out / "old_all_raw.jsonl", encoding="utf-8")]
old_texts = [f"{r['q']} {r['a']}" for r in old if r["status"] == "ok"]
K = len(old_texts)
res = {"matched_size": K, "old": matched(old_texts, K, 1)}
cfg = load_config("configs/dev_0p5b.yaml")
nb = Notebook("results/phase1/notebook.jsonl")
notes = nb.active()
tok, model = load_model(cfg.model.name)
cfg.dream.mix = {"qa": 1.0}
cfg.dream.samples_per_note = 3
for temp in (0.5, 0.8, 1.0):
    seed_everything(0)
    cfg.dream.qa_temp = temp
    t0 = time.time()
    samples, st = D.generate_candidates(model, tok, notes, cfg.dream, D.DreamState(), 1.0, 0)
    texts = [f"{s['user']} {s['assistant']}" for s in samples]
    res[f"new_temp{temp}"] = {"n_valid": len(samples), "first_pass_yield": st["first_pass_yield"],
                              "final_fill": st["final_fill"], "yield_per_attempt": st["yield_per_attempt"],
                              "seconds": time.time() - t0, "rejected": st["rejected"]["qa"],
                              "full": {"distinct_2": distinct_n(texts, 2), "self_bleu": self_bleu(texts)},
                              f"matched{K}": matched(texts, K)}
    print(temp, json.dumps(res[f"new_temp{temp}"]), flush=True)
print("old", res["old"])
write_json(out / "stepA_diversity.json", res)
