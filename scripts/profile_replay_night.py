"""Reproduce a real replay night's TRAINING (day-5 dreams of both_v2 + reservoir/old-skill mix, as in the pipeline) and log
RSS / torch reserved+allocated / device-used memory every second.
Usage: python scripts/profile_replay_night.py <lean:0|1> <out.csv> [epochs]"""
import sys, time, threading, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import psutil, torch, pandas as pd
from sleepllm.config import Config, load_config
from sleepllm.llm import load_model
from sleepllm import sleep as S
from sleepllm.replay import ReplayBuffer, load_old_skill
from sleepllm.utils import read_jsonl

lean = sys.argv[1] == "1"; out = sys.argv[2]; epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
cfg = load_config("configs/dev_0p5b.yaml", ["lora.lr=0.0001", f"lora.epochs={epochs}", f"lora.lean_logits={'true' if lean else 'false'}"])
proc = psutil.Process()
tok, model = load_model(cfg.model.name); model = S.attach_adapter(model, cfg.lora)
R = "results/phase2_pilot/both_v2"
dreams = read_jsonl(f"{R}/dreams_day5.jsonl"); st = json.load(open(f"{R}/state.json", encoding="utf-8"))
rb = ReplayBuffer(load_old_skill(400), 1500, 0); rb.reservoir, rb.seen = st["reservoir"], st["seen"]
past_old, info = rb.sample_mix(len(dreams), 0.4, 0.2)
batch = dreams + [{**r, "replay": True} for r in past_old]
lens = [len(S._encode(tok, x, 256)[0]) for x in batch]
print(f"lean_logits={lean} n={len(batch)} (new {len(dreams)} + replay {len(past_old)}) max_len={max(lens)} mean_len={sum(lens)/len(lens):.0f}", flush=True)
rows = []; stop = threading.Event(); t0 = time.time()
def samp():
    while not stop.wait(1.0):
        free, total = torch.cuda.mem_get_info()
        rows.append(dict(t=round(time.time() - t0, 1), rss=proc.memory_info().rss / 2**20, reserved=torch.cuda.memory_reserved() / 2**20,
                         allocated=torch.cuda.memory_allocated() / 2**20, dev_used=(total - free) / 2**20,
                         sys_free_gb=psutil.virtual_memory().available / 2**30))
threading.Thread(target=samp, daemon=True).start()
res = S.train(model, tok, batch, cfg, 5)
stop.set(); time.sleep(1.2)
df = pd.DataFrame(rows); df.to_csv(out, index=False)
print(json.dumps({"train_s": round(res["train_seconds"]), "steps": res["steps"], "loss_first": round(res["loss_first"], 3), "loss_last": round(res["loss_last"], 3),
                  "peak_rss_mb": round(df.rss.max()), "rss_median_mb": round(df.rss.median()), "peak_reserved_mb": round(df.reserved.max()),
                  "peak_allocated_mb": round(df.allocated.max()), "peak_device_used_mb": round(df.dev_used.max()), "min_sys_free_gb": round(df.sys_free_gb.min(), 2)}))
