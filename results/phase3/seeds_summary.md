## 3-seed summary, day 5 (mean +- sample std; seed also changes the world)

| metric | naive | both_v2 (S1) |
|---|---|---|
| acc today | 0.108 +- 0.064 (n=3) | 0.495 +- 0.127 (n=3) |
| acc earlier | 0.035 +- 0.009 (n=3) | 0.367 +- 0.046 (n=3) |
| acc earlier (mean d2-5) | 0.059 +- 0.015 (n=3) | 0.351 +- 0.017 (n=3) |
| old-skill | 0.392 +- 0.018 (n=3) | 0.458 +- 0.020 (n=3) |
| WikiText ppl | 44.2 +- 3.8 (n=3) | 48.8 +- 3.4 (n=3) |
| matched distinct-2 | 0.375 +- 0.031 (n=3) | 0.452 +- 0.019 (n=3) |
| peak RSS MB | 2114 +- 69 (n=2) | 2492 +- 712 (n=3) |
| night s (mean) | 287 +- 161 (n=3) | 1112 +- 381 (n=3) |

### per seed (acc today / acc earlier / old-skill / ppl / matched d2 / alarm days / peak RSS MB)
- naive seed 0: 0.182 / 0.025 / 0.375 / 46.2 / 0.404 / 0 / nan
- naive seed 1: 0.077 / 0.043 / 0.390 / 46.5 / 0.378 / 0 / 2163
- naive seed 2: 0.066 / 0.036 / 0.410 / 39.9 / 0.342 / 4 / 2066
- both seed 0: 0.500 / 0.408 / 0.440 / 52.5 / 0.433 / 1 / 2140
- both seed 1: 0.365 / 0.317 / 0.455 / 47.9 / 0.452 / 0 / 2025
- both seed 2: 0.618 / 0.375 / 0.480 / 46.0 / 0.471 / 2 / 3311

### go / no-go (criteria from PLAN.md, unchanged)
- [FAIL] 1 peak RSS < 3 GB in the replay arm: max over seeds 3311 MB
- [PASS] 2 mean acc earlier >= 0.20 and acc today >= 0.25 (day 5): earlier 0.367, today 0.495
- [FAIL] 3 old-skill >= 0.43 and ppl <= 40 (day 5, mean): old-skill 0.458, ppl 48.8
- [FAIL] 4 matched distinct-2 both >= naive AND no alarm in >= 2 of 3 seeds: distinct-2 both 0.452 vs naive 0.375; seeds without alarm 1/3
- [PASS] 5 no run died of an unexplained crash in Phase 3: all Phase 3 runs completed (set by hand: see LOG.md)
- [PASS] 6 per-night time measured (extrapolation reported): mean night 1112 s -> 30 nights = 9.3 h of night time, excluding evaluation

**Decision: NO-GO**; failing: 1, 3, 4