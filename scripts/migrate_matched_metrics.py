"""One-off: add size-matched diversity columns (k=100) to results/phase2_pilot/metrics.csv for rows that lack them,
recomputed from the saved dreams_day*.jsonl (same embedder/seed as the live pipeline)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sleepllm.diversity import default_embedder, matched_metrics  # noqa: E402
from sleepllm.utils import read_jsonl  # noqa: E402

root = Path("results/phase2_pilot")
df = pd.read_csv(root / "metrics.csv")
cols = ["distinct_2_m", "self_bleu_m", "embedding_dispersion_m"]
for c in cols:
    if c not in df:
        df[c] = np.nan
emb = default_embedder()
for i, r in df.iterrows():
    if not pd.isna(r["distinct_2_m"]):
        continue
    s = read_jsonl(root / r["arm"] / f"dreams_day{int(r['day'])}.jsonl")
    t = [f"{x['user']} {x['assistant']}" for x in s]
    m = matched_metrics(t, np.asarray(emb(t)), k=100)
    for c in cols:
        df.loc[i, c] = m[c]
df.to_csv(root / "metrics.csv", index=False)
print(df.groupby("arm")[cols].mean().round(3))
