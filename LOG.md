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
