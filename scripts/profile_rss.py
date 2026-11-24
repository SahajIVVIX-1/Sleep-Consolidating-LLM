"""Profile process RSS per pipeline step (peak during each step, sampled every 0.1 s).
Runs day 1 of one pilot arm in a scratch folder. Usage: python scripts/profile_rss.py [arm] [out_json]"""
import functools
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import psutil  # noqa: E402
import torch  # noqa: E402

proc = psutil.Process()
rows = []
cur = {"peak": 0.0}
stop = threading.Event()


def sampler():
    while not stop.wait(0.1):
        cur["peak"] = max(cur["peak"], proc.memory_info().rss / 2**20)


threading.Thread(target=sampler, daemon=True).start()
depth = [0]


def wrap(mod, name, label=None):
    fn = getattr(mod, name)
    label = label or f"{mod.__name__.split('.')[-1]}.{name}"

    @functools.wraps(fn)
    def inner(*a, **k):
        r0 = proc.memory_info().rss / 2**20
        saved = cur["peak"]
        cur["peak"] = r0
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        d = depth[0]
        depth[0] += 1
        try:
            return fn(*a, **k)
        finally:
            depth[0] -= 1
            r1 = proc.memory_info().rss / 2**20
            rows.append({"step": "  " * d + label, "rss_before": round(r0), "rss_after": round(r1),
                         "peak_during": round(max(cur["peak"], r1)),
                         "vram_reserved_peak": round(torch.cuda.max_memory_reserved() / 2**20), "seconds": round(time.time() - t0, 1)})
            cur["peak"] = max(saved, cur["peak"])
    setattr(mod, name, inner)


import sleepllm.dream as D  # noqa: E402
import sleepllm.diversity as DV  # noqa: E402
import sleepllm.eval as EV  # noqa: E402
import sleepllm.llm as L  # noqa: E402
import sleepllm.pilot as P  # noqa: E402
import sleepllm.replay as R  # noqa: E402
import sleepllm.sleep as S  # noqa: E402
from sleepllm.config import load_config  # noqa: E402

arm = sys.argv[1] if len(sys.argv) > 1 else "replay"
outj = sys.argv[2] if len(sys.argv) > 2 else "results/profile_rss.json"
for mod, names in [(EV, ["load_probe", "eval_old_skills", "wikitext_ppl", "eval_new_knowledge"]),
                   (L, ["load_model"]), (S, ["attach_adapter", "train"]), (R, ["load_old_skill"]),
                   (D, ["make_dreams", "generate_candidates"]), (DV, ["default_embedder", "night_metrics", "dedupe",
                                                                      "select_diverse", "embedding_dispersion"])]:
    for n in names:
        wrap(mod, n)
# modules that imported names directly
P.load_model, P.load_old_skill, P.sleep_mod, P.ev = L.load_model, R.load_old_skill, S, EV
D.__dict__["default_embedder"] = DV.default_embedder

import json  # noqa: E402
import shutil  # noqa: E402

scratch = Path("results/_profile_scratch")
shutil.rmtree(scratch, ignore_errors=True)
(scratch).mkdir(parents=True)
shutil.copytree("results/phase2_pilot/shared", scratch / "shared")  # reuse notes/baseline: no base-model extraction
cfg = load_config("configs/dev_0p5b.yaml", [f"out_dir={scratch}"])
rows.append({"step": "start (imports done)", "rss_before": round(proc.memory_info().rss / 2**20), "rss_after": 0,
             "peak_during": 0, "seconds": 0})
P.run_pilot(cfg, days=1, arms=[arm])
stop.set()
Path(outj).write_text(json.dumps(rows, indent=1), encoding="utf-8")
print("\n".join(f"{r['step']:<38} before {r['rss_before']:>5}  peak {r['peak_during']:>5}  after {r['rss_after']:>5}  vram_res {r.get('vram_reserved_peak', 0):>5}  {r['seconds']:>6}s"
                for r in rows))
shutil.rmtree(scratch, ignore_errors=True)
