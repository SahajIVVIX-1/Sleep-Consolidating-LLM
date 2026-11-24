"""RSS trace of the real sleep.train() on real day-1 dreams + replay items (epochs=1)."""
import sys, time, threading, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import psutil, torch
from sleepllm.config import Config
from sleepllm.llm import load_model
from sleepllm import sleep as S
from sleepllm.replay import ReplayBuffer, load_old_skill
from sleepllm.utils import read_jsonl

proc = psutil.Process(); mb = lambda: proc.memory_info().rss / 2**20
cfg = Config(); cfg.lora.epochs = 1
tok, model = load_model(cfg.model.name); model = S.attach_adapter(model, cfg.lora)
dreams = read_jsonl("results/phase2_pilot/replay/dreams_day1.jsonl")
rb = ReplayBuffer(load_old_skill(400), 300, 0)
samples = dreams + rb.sample(len(dreams), 0.3)
print("n samples", len(samples), "start rss", round(mb()), flush=True)
cur = {"step": 0}; peak = [0.0]; stop = threading.Event(); trace = []
def samp():
    t0 = time.time()
    while not stop.wait(0.25):
        r = mb(); peak[0] = max(peak[0], r); trace.append((round(time.time() - t0, 1), cur["step"], round(r)))
threading.Thread(target=samp, daemon=True).start()
st = S.train(model, tok, samples, cfg, 0, log=lambda r: (cur.__setitem__("step", r["step"]), print("step", r["step"], "rss", round(mb()), "peak", round(peak[0]), flush=True)))
stop.set()
print("train", round(st["train_seconds"]), "s; peak rss", round(peak[0]), "MB; rss after", round(mb()))
big = max(trace, key=lambda x: x[2]); print("max trace point (t, step, rss):", big)
