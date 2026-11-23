"""Night only: dream from out_dir/notebook.jsonl, train adapter, save to out_dir/adapter."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sleepllm.config import load_config  # noqa: E402
from sleepllm.llm import load_model  # noqa: E402
from sleepllm.notebook import Notebook  # noqa: E402
from sleepllm.pipeline import dream_diversity, night_dream_stage, night_train_stage  # noqa: E402
from sleepllm.utils import seed_everything, write_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/dev_0p5b.yaml")
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
cfg = load_config(a.config, a.set)
seed_everything(cfg.seed)
out = Path(cfg.out_dir)
nb = Notebook(out / "notebook.jsonl")
tok, model = load_model(cfg.model.name, cfg.model.load_4bit)
samples, dstats = night_dream_stage(cfg, model, tok, nb, out)
_, tstats = night_train_stage(cfg, model, tok, samples, out)
write_json(out / "night.json", {"dream": dstats, "train": tstats, "diversity": dream_diversity(samples)})
print(dstats, tstats)
