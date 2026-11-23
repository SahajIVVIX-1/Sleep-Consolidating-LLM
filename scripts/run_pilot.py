"""Phase 2 pilot. Usage: python scripts/run_pilot.py [--days 5] [--arms naive reg replay both] [--set a.b=c ...]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sleepllm.config import load_config  # noqa: E402
from sleepllm.pilot import run_pilot  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/dev_0p5b.yaml")
ap.add_argument("--days", type=int, default=5)
ap.add_argument("--arms", nargs="*", default=None)
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
cfg = load_config(a.config, ["out_dir=results/phase2_pilot"] + a.set)
run_pilot(cfg, a.days, a.arms)
