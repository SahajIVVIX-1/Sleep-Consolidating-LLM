"""Phase 1: one full day/night cycle. Usage: python scripts/run_phase1.py [--config configs/dev_0p5b.yaml] [--set a.b=c]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sleepllm.config import load_config  # noqa: E402
from sleepllm.pipeline import run_phase1  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/dev_0p5b.yaml")
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
M = run_phase1(load_config(a.config, a.set))
print(json.dumps(M, indent=1)[:6000])
