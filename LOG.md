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
