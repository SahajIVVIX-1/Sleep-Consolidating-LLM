"""Trace RSS and CUDA memory inside a few training micro-steps (forward / backward / optimizer)."""
import sys, time, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import psutil, torch
from sleepllm.config import Config
from sleepllm.llm import load_model
from sleepllm import sleep as S
from sleepllm.utils import read_jsonl

proc = psutil.Process()
mb = lambda: proc.memory_info().rss / 2**20
cfg = Config()
tok, model = load_model(cfg.model.name)
model = S.attach_adapter(model, cfg.lora)
samples = read_jsonl("results/phase2_pilot/replay/dreams_day1.jsonl")[:64]
enc = [S._encode(tok, s, 256) for s in samples]
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=2e-4)
peak = [0.0]; stop = threading.Event()
def samp():
    while not stop.wait(0.05): peak[0] = max(peak[0], mb())
threading.Thread(target=samp, daemon=True).start()
def mark(label, t0):
    torch.cuda.synchronize()
    print(f"{label:<22} rss {mb():7.0f} MB  peak-since-last {peak[0]:7.0f}  cuda alloc {torch.cuda.memory_allocated()/2**20:6.0f}  reserved {torch.cuda.memory_reserved()/2**20:6.0f}  {time.time()-t0:5.1f}s", flush=True)
    peak[0] = mb()
print("start rss", round(mb())); model.train(); peak[0] = mb()
for step in range(4):
    for k in range(3):
        t0 = time.time()
        ids, lab, att = S._batch(tok, enc[(step*3+k)*2:(step*3+k)*2+2]); mark(f"s{step}.{k} batch", t0); t0 = time.time()
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float(); mark(f"s{step}.{k} forward", t0); t0 = time.time()
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1), ignore_index=-100); mark(f"s{step}.{k} loss", t0); t0 = time.time()
        loss.backward(); mark(f"s{step}.{k} backward", t0)
        del logits, loss
    t0 = time.time(); opt.step(); opt.zero_grad(); mark(f"s{step} opt.step", t0)
stop.set()
