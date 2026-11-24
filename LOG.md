# LOG

## 2026-11-23/24 Phase 0
- Env: Windows 11, Python 3.12.10 (user chose 3.12 over 3.11), RTX 3050 Laptop 4096 MiB, no WSL2 needed.
- torch 2.6.0+cu124 install took ~25 min (slow link, ~1.2 MB/s). cuda_available=True.
- Unit tests (notebook, world, diversity): 15 passed.
- bitsandbytes 0.50.2 4-bit NF4 load on native Windows: **works**. transformers 5.24.0, peft 0.22.0.
- Qwen2.5-0.5B-Instruct 4-bit: VRAM after load 436 MB, peak during generate 448 MB (results/phase0.json). First load 870 s was the model download; cached load 1.3 s.
- 20-step QLoRA on random tokens (r=16, all linear, grad ckpt):
  - batch 4 x seq 384: 60.7 s, peak 3616 MB allocated / 3828 reserved -> too close to 4 GB.
  - batch 2 x seq 256: 12.6 s, peak 1748 MB allocated / 2044 reserved (results/phase0_b2s256.json).
  - Decision: default micro_batch=2, grad_accum=8, max_seq_len=256. (Large 152k vocab logits dominate memory.)
  - Loss ~13 -> 12 is on random tokens, only a smoke test.
- Note: requirements.txt is a pip freeze; torch needs the cu124 index URL (noted in file header).

## Phase 1 (one day/night cycle, seed 0, Qwen2.5-0.5B-Instruct 4-bit, results/phase1/)
- Ran `scripts/run_phase1.py` once, end to end. Raw outputs in results/phase1/ (metrics.json, qa_*.jsonl, dreams.jsonl, notebook.jsonl, train_log.csv).
- Broke: first extraction prompt parsed 5/50 sentences (model prefixes "Fact:", wrong relations). Fixed by stripping the prefix and one few-shot example per relation -> 41/50 parsed, recall 0.70, precision 0.85 vs ground truth.
- Dream QA yield is low: 30 of 150 raw QA samples passed the answer-in-note filter (20%); paraphrase 86/100; counterfactual 25/50.
- QA accuracy (250 questions = 50 facts x 5 phrasings): before 0.000, after night (no notebook) 0.144 (held-out 0.130), with notebook (base) 0.728.
- Old-skill probe overall 0.460 -> 0.415 (n=200, CIs overlap); GSM8K 0.20 -> 0.10 (n=40); WikiText ppl 26.57 -> 28.83.
- Night: 178 s (dream 58 s + train 119 s), 27 optimiser steps, peak 1.4 GB VRAM.
- Not done: ablation without counterfactuals; multiple seeds.

## Phase 2, Step A: dream-QA yield (results/phase2_stepA/)
- Diagnosis of old generator (150 raw QA at temp 0.8, no retries): ok 29 (19.3%), format 78, ungrounded 32, off_topic 8, truncated 2, duplicate 1. In the 50-sample of rejects: format 33, ungrounded 15, off_topic 2. Typical failures: the 0.5B model invents long elaborate answers ("A: The name Kriader is derived from ...") or asks about things the note does not say, so answer >14 words or answer not in note.
- First attempt at fixing (strict Q:/A: format, 2 few-shot examples, temp 0.5, grounding check, 2 retries): first-pass yield only 30%, final fill 53% (format 136 rejects, truncated 46). Format constraints alone were not enough.
- Fix that worked: **answer-first generation**. The answer span is taken from the note's extracted key (its object) and the model only writes the question; validation requires the question to mention the subject, not contain the answer, end with '?'. Consequence: only notes with a parsed key (41/50 on day 1) get QA dreams; the rest get paraphrase/counterfactual.
- Result (150 slots, qa temp 0.5): first-pass yield 68.7%, final fill 88% after <=2 retries, valid per generation attempt 57%. Old: 19.3%. Generation time 21 s (old: 30 s for 150 raw). Rejects remaining: off_topic 55, duplicate 42, format 1.
- Diversity (size-matched to the old 29 valid items, mean of 20 subsamples): distinct-2 0.634 vs old 0.503, self-BLEU 0.263 vs 0.469 (lower is better), embedding dispersion 0.743 vs 0.764 (slightly lower). On the full 132-item set distinct-2 is 0.32 simply because bigram repeats grow with set size. Qa temp sweep (0.5/0.8/1.0): yield stays 69-70%, matched-size distinct-2 0.63/0.65/0.66. Kept qa_temp=0.5 as specified.
- Caveat: "no loss in diversity" holds for distinct-n and self-BLEU; dispersion is ~0.02 lower (one old set, no variance estimate).

## Phase 2, Step D: 5-day pilot, 4 arms (results/phase2_pilot/, plots in results/phase2_pilot/plots/)
Qwen2.5-0.5B-Instruct 4-bit, seed 0, same world and same base-model notes for all arms, one adapter trained continuously over 5 nights, QA scored on held-out phrasings without notebook (n_today 66-100 per day, n_earlier up to 282). Base model: old-skill 0.460, WikiText ppl 26.57. ONE SEED, 5 DAYS: indicative only; with n~70-100 questions per cell, differences below ~0.1 are within noise.

| metric (day 1/2/3/4/5) | naive | +reg | +replay | +both |
|---|---|---|---|---|
| acc today (no notebook) | .38/.15/.23/.22/.18 | .19/.24/.21/.22/.11 | .37/.36/.35/.41/.36 | .27/.21/.20/.26/.26 |
| acc earlier days | -/.05/.05/.04/.03 | -/.09/.11/.05/.03 | -/.15/.11/.03/.07 | -/.13/.07/.06/.06 |
| old-skill overall | .425/.400/.390/.385/.375 | .390/.420/.425/.410/.390 | .450/.470/.440/.445/.465 | .405/.450/.440/.435/.450 |
| WikiText ppl | 31.1/37.2/41.8/46.5/46.2 | 31.2/40.2/46.6/54.5/62.5 | 30.4/34.3/34.9/39.7/37.0 | 30.8/36.3/40.7/42.6/43.5 |
| dream distinct-2 | .36/.39/.42/.38/.38 | .39/.36/.35/.32/.29 | .36/.34/.35/.37/.35 | .39/.34/.35/.34/.33 |
| dream embedding dispersion | .75/.76/.77/.77/.77 | .76/.76/.76/.74/.75 | .75/.75/.76/.75/.76 | .76/.76/.76/.75/.72 |
| night time day 1..5 (s) | 305/806/529/427/290 | 885/795/558/691/767 | 504/383/302/257/237 | 625/530/350/404/368 |

- Collapse alarm fired (distinct-2 > 15% below night 1): reg day 4 and 5, both day 5; naive/replay never (alarms only act in regulariser arms; temperature bump reached +0.2 and +0.1).
- Peak VRAM: naive 1.33 GB, reg 1.34 GB, replay 2.08 GB, both 2.08 GB (replay mixes in longer sequences).
- Peak process RAM (RSS): naive/reg NOT measured (run before the sampler existed); replay 5.40 GB on day 2 in the OOM-killed run (pre-fix), 2.76 GB on the resumed days 4-5; both 5.48 GB on day 2 (4.59 GB day 3, <=3.4 GB other days), min free system RAM 1.44 GB. The `_free()` before training did NOT remove the day 1-3 RAM spike. Isolated profiling (scripts/profile_*.py) showed it is inside sleep.train; the first forward adds ~0.6 GB of CUDA/bitsandbytes kernel loading, but a standalone train() peaked at only 2.2 GB, so the full 4.8-5.5 GB is unexplained (hypothesis, unverified: VRAM spill to system RAM on a 4 GB card shared with other apps).
- Incidents: pilot was OOM-killed by the memory reaper twice (replay arm; then a profiling run) while another project was using several GB. Final runs were launched detached via scripts/run_remaining.bat and completed without a kill.
- Not done: ablation of the optional KL term; multiple seeds; the 30-day run.

## Phase 2 round 2 (v2): regulariser fix (A) and cross-day retention (B)  [results/phase2_pilot/*_v2, plots in results/phase2_pilot/plots/]
Same seed / world / notes as the first pilot; 5 days; one seed => indicative only. n_earlier = 282 questions on day 5 (95% CI about +-0.05).

**A. Why reg looked bad.** (1) Full-set distinct-2 is size-dependent; at matched size (k=100) naive and reg were 0.450 vs 0.440 averaged over days. (2) The pool was too small: day 2 generated 626 valid candidates, dedupe (cos>0.92) removed 298, 328 remained for a target of 294 -> selection barely selected. (3) When QA/multihop were scarce the quota relaxation topped the set up with counterfactuals (35-40% of the set on days 4-5, distinct-2 0.27-0.31 because of the shared "In an alternate world" prefix). (4) The collapse alarm used the size-confounded metric.
Fixes: adaptive generation until each kind's deduped pool >= 2x quota (max 3 rounds), relaxation capped at 1.5x quota, alarm on size-matched distinct-2/dispersion, per-night log of candidates / dedupe-removed / selected / rejected and kept-vs-rejected diversity (rejected_day*.jsonl saved), 10 new unit tests (threshold side, scale invariance, outlier preference, quota cap, coverage, matched-size metric).

**B. Retention.** replay v2 = batch of 40% past-day dreams (day-balanced reservoir, counterfactuals excluded, dreams of superseded facts pruned) + 20% old-skill + 40% new; LR 2e-4 -> 1e-4 (3 epochs). One cumulative adapter kept, not merge-per-night: merging into a 4-bit base needs dequantise/requantise (lossy, extra RAM/VRAM on a 4 GB card) and a fresh rank-16 adapter each night forgets nothing by itself but would need the same rehearsal anyway; capacity of one r=16 adapter is the thing to watch at 30 days.

| day-5 / mean | naive | reg | reg_v2 | replay | replay_v2 | both | both_v2 |
|---|---|---|---|---|---|---|---|
| acc earlier days, day 5 | .025 | .025 | .043 | .071 | .195 | .060 | **.206** |
| acc earlier days, mean days 2-5 | .042 | .066 | .040 | .090 | .207 | .079 | .216 |
| acc today, day 5 | .182 | .106 | .167 | .364 | .227 | .258 | .212 |
| old-skill, day 5 (base .460) | .375 | .390 | .385 | .465 | .450 | .450 | .425 |
| WikiText ppl, day 5 (base 26.6) | 46.2 | 62.5 | 61.1 | 37.0 | 35.6 | 43.5 | 40.7 |
| dream distinct-2 (matched k=100), day 5 | .404 | .364 | .379 | .413 | .420 | .399 | .471 |
| matched distinct-2 days 1..5 | .47 .50 .47 .42 .40 | .51 .50 .43 .41 .36 | .52 .52 .50 .45 .38 | | | .51 .46 .43 .43 .40 | .52 .52 .49 .48 .47 |
| collapse alarm fired | - | d4,d5 (old metric) | d5 | - | - | d5 (old metric) | none |
| peak RSS (MB) / min free RAM (GB) | not measured | not measured | 2032 / 3.6 | 5395 / 0.29 (v1), 5883 / 0.37 (v2 incl. killed attempts) | | 5480 / 1.44 | 5494 / 0.26 |
| peak VRAM (GB) | 1.33 | 1.34 | 1.34 | 2.08 | 2.08 | 2.08 | 2.08 |

Targets: (A) distinct-2 >= naive and no alarm: reg_v2 yes on days 1-4, no on day 5 (0.379 vs 0.404) and the alarm fired on day 5 -> NOT fully met; both_v2 met it (0.47-0.52 vs naive 0.40-0.50, no alarm). (B) acc_earlier >= 0.2 at day 5: both_v2 .206 met; replay_v2 .195 missed by 0.005 (within noise of the target).
Trade-offs seen: acc on today's facts fell vs v1 replay (.364 -> .227), the lower LR and the smaller share of new dreams are the likely reason (not isolated: no run changes only one factor; v2 replay arms differ from v1 in LR, mix and pruning together).
Not fixed: reg-type arms still show high WikiText ppl (61 at day 5). reg_v2 trains on more dream samples per night than naive (192 vs 119 on day 5) - untested confound.

**Crash investigation (replay_v2 night 3, both_v2 day 1).** Deaths of the visible-window runs left no trace. With faulthandler on, two both_v2 deaths were access violations (0xc0000005, python312.dll offset 0x27e990) exactly 10:01 after start, each log starting with "Timeout (0:10:00)!" = my own faulthandler.dump_traceback_later watchdog; removing it let both_v2 run all 5 days. My first diagnosis (bitsandbytes fused 4-bit GEMM kernel) was wrong and was reverted to opt-in (SLEEPLLM_DISABLE_FUSED_4BIT_GEMM=1; not used for any result). The three earlier silent deaths are NOT explained. See results/phase2_pilot/crash_evidence/README.md. Observed but not investigated: the RAM spikes (5.4-5.9 GB RSS) appear only in arms with replay; reg_v2 peaked at 2.0 GB.
