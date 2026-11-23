"""One plot per metric from results/phase2_pilot/metrics.csv (+ shared/baseline.json for day-0 reference lines)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase2_pilot")
df = pd.read_csv(root / "metrics.csv")
base = json.loads((root / "shared" / "baseline.json").read_text(encoding="utf-8"))
ref = {"old_overall": base["old_skill"]["overall"]["acc"], "old_arc": base["old_skill"]["arc_easy"]["acc"],
       "old_hellaswag": base["old_skill"]["hellaswag"]["acc"], "old_gsm8k": base["old_skill"]["gsm8k"]["acc"],
       "wikitext_ppl": base["wikitext_ppl"]}
metrics = ["acc_today", "acc_earlier", "acc_all", "old_overall", "old_arc", "old_hellaswag", "old_gsm8k", "wikitext_ppl",
           "distinct_1", "distinct_2", "self_bleu", "embedding_dispersion", "vocab_entropy", "entropy_ratio",
           "first_pass_yield", "night_s", "dream_s", "train_s", "peak_vram_night_mb"]
out = root / "plots"
out.mkdir(exist_ok=True)
colors = {"naive": "#d95f02", "reg": "#1b9e77", "replay": "#7570b3", "both": "#e7298a"}
for m in metrics:
    if m not in df or df[m].replace("", float("nan")).dropna().empty:
        continue
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for arm, g in df.groupby("arm"):
        g = g.sort_values("day")
        ax.plot(g["day"], pd.to_numeric(g[m], errors="coerce"), marker="o", label=arm, color=colors.get(arm))
    if m in ref:
        ax.axhline(ref[m], ls="--", color="gray", lw=1, label="base model (day 0)")
    ax.set_xlabel("day")
    ax.set_ylabel(m)
    ax.set_xticks(sorted(df["day"].unique()))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"{m}.png", dpi=120)
    plt.close(fig)
print("wrote", len(list(out.glob("*.png"))), "plots to", out)
