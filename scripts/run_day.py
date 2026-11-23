"""Day only: generate world day N, read documents, write notes. Usage: run_day.py [--day 1]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sleepllm.config import load_config  # noqa: E402
from sleepllm.llm import load_model  # noqa: E402
from sleepllm.pipeline import day_stage, make_day  # noqa: E402
from sleepllm.utils import seed_everything, write_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/dev_0p5b.yaml")
ap.add_argument("--day", type=int, default=1)
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
cfg = load_config(a.config, a.set)
seed_everything(cfg.seed)
out = Path(cfg.out_dir)
out.mkdir(parents=True, exist_ok=True)
_, dd = make_day(cfg, a.day)
tok, model = load_model(cfg.model.name, cfg.model.load_4bit)
stats, _ = day_stage(cfg, model, tok, dd, out)
write_json(out / f"day{a.day}_extraction.json", stats)
print({k: v for k, v in stats.items() if k != "raw_examples"})
