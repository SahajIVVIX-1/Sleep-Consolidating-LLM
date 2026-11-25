"""Step C: mean +- std over seeds (day 5) for naive vs both_v2 (S1 setting), and the go/no-go check from PLAN.md.

Seed 0: naive = phase-2 pilot run, both_v2 = Phase-3 sweep S1 (same settings as seeds 1-2: lr 2e-4, past 40% / old 20%).
Seeds 1,2: results/phase3/seed{N}. Usage: python scripts/aggregate_seeds.py [out.md]
The criteria below are copied from PLAN.md ("Go / no-go for the 30-day run") and must not be edited after the fact."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results")
SRC = {  # (seed, arm) -> (metrics csv, arm name in csv, done.json)
    (0, "naive"): (R / "phase2_pilot/metrics.csv", "naive", R / "phase2_pilot/naive/done.json"),
    (0, "both"): (R / "phase3/sweep_S1/metrics.csv", "both_v2", R / "phase3/sweep_S1/both_v2/done.json"),
}
for s in (1, 2):
    SRC[(s, "naive")] = (R / f"phase3/seed{s}/metrics.csv", "naive", R / f"phase3/seed{s}/naive/done.json")
    SRC[(s, "both")] = (R / f"phase3/seed{s}/metrics.csv", "both_v2", R / f"phase3/seed{s}/both_v2/done.json")


def load(seed, arm):
    csv, name, done = SRC[(seed, arm)]
    if not csv.exists():
        return None
    d = pd.read_csv(csv)
    d = d[d.arm == name].sort_values("day")
    if len(d) < 5:
        return None
    d5 = d[d.day == 5].iloc[0]
    j = json.loads(done.read_text()) if done.exists() else {}
    return {
        "acc_today": d5.acc_today, "acc_earlier": d5.acc_earlier, "old_skill": d5.old_overall, "ppl": d5.wikitext_ppl,
        "distinct2_m": d5.distinct_2_m, "mean_distinct2_m": d[d.day >= 1].distinct_2_m.mean(),
        "mean_acc_earlier_d2_5": d[d.day >= 2].acc_earlier.mean(), "alarm_days": int(d.alarm.notna().sum()),
        "night_s": d.night_s.mean(), "peak_rss_mb": j.get("peak_rss_mb", np.nan), "peak_vram_mb": j.get("peak_vram_mb", np.nan),
        "n_dreams_d5": d5.n_dreams,
    }


rows = {a: {s: load(s, a) for s in (0, 1, 2)} for a in ("naive", "both")}
cols = [("acc_today", "acc today"), ("acc_earlier", "acc earlier"), ("mean_acc_earlier_d2_5", "acc earlier (mean d2-5)"),
        ("old_skill", "old-skill"), ("ppl", "WikiText ppl"), ("distinct2_m", "matched distinct-2"),
        ("peak_rss_mb", "peak RSS MB"), ("night_s", "night s (mean)")]


def ms(arm, key):
    v = np.array([rows[arm][s][key] for s in (0, 1, 2) if rows[arm][s] is not None], float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "n/a"
    sd = v.std(ddof=1) if len(v) > 1 else float("nan")
    f = "{:.0f}" if key in ("peak_rss_mb", "night_s") else ("{:.1f}" if key == "ppl" else "{:.3f}")
    return f"{f.format(v.mean())} +- {f.format(sd) if len(v) > 1 else 'n/a'} (n={len(v)})"


out = ["## 3-seed summary, day 5 (mean +- sample std; seed also changes the world)", "",
       "| metric | naive | both_v2 (S1) |", "|---|---|---|"]
for k, label in cols:
    out.append(f"| {label} | {ms('naive', k)} | {ms('both', k)} |")
out += ["", "### per seed (acc today / acc earlier / old-skill / ppl / matched d2 / alarm days / peak RSS MB)"]
for a in ("naive", "both"):
    for s in (0, 1, 2):
        r = rows[a][s]
        out.append(f"- {a} seed {s}: " + ("not finished" if r is None else
                   f"{r['acc_today']:.3f} / {r['acc_earlier']:.3f} / {r['old_skill']:.3f} / {r['ppl']:.1f} / {r['distinct2_m']:.3f} / "
                   f"{r['alarm_days']} / {r['peak_rss_mb']:.0f}"))

# ---- go / no-go, exactly as in PLAN.md
bs = [rows["both"][s] for s in (0, 1, 2)]
ns = [rows["naive"][s] for s in (0, 1, 2)]
if all(x is not None for x in bs + ns):
    m = lambda xs, k: float(np.mean([x[k] for x in xs]))  # noqa: E731
    c = []
    rss = max(x["peak_rss_mb"] for x in bs)
    c.append(("1 peak RSS < 3 GB in the replay arm", rss < 3000, f"max over seeds {rss:.0f} MB"))
    c.append(("2 mean acc earlier >= 0.20 and acc today >= 0.25 (day 5)", m(bs, "acc_earlier") >= 0.20 and m(bs, "acc_today") >= 0.25,
              f"earlier {m(bs, 'acc_earlier'):.3f}, today {m(bs, 'acc_today'):.3f}"))
    c.append(("3 old-skill >= 0.43 and ppl <= 40 (day 5, mean)", m(bs, "old_skill") >= 0.43 and m(bs, "ppl") <= 40,
              f"old-skill {m(bs, 'old_skill'):.3f}, ppl {m(bs, 'ppl'):.1f}"))
    no_alarm = sum(1 for x in bs if x["alarm_days"] == 0)
    c.append(("4 matched distinct-2 both >= naive AND no alarm in >= 2 of 3 seeds",
              m(bs, "distinct2_m") >= m(ns, "distinct2_m") and no_alarm >= 2,
              f"distinct-2 both {m(bs, 'distinct2_m'):.3f} vs naive {m(ns, 'distinct2_m'):.3f}; seeds without alarm {no_alarm}/3"))
    c.append(("5 no run died of an unexplained crash in Phase 3", True, "all Phase 3 runs completed (set by hand: see LOG.md)"))
    c.append(("6 per-night time measured (extrapolation reported)", True,
              f"mean night {m(bs, 'night_s'):.0f} s -> 30 nights = {30 * m(bs, 'night_s') / 3600:.1f} h of night time, excluding evaluation"))
    out += ["", "### go / no-go (criteria from PLAN.md, unchanged)"]
    out += [f"- [{'PASS' if ok else 'FAIL'}] {name}: {info}" for name, ok, info in c]
    out.append(f"\n**Decision: {'GO' if all(ok for _, ok, _ in c) else 'NO-GO'}**; failing: "
               + (", ".join(n.split(' ')[0] for n, ok, _ in c if not ok) or "none"))
else:
    out += ["", "go/no-go: not computed (some runs unfinished)"]
text = "\n".join(out)
print(text)
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(text, encoding="utf-8")
