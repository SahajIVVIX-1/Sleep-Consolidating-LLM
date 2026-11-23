# PLAN.md — Sleep-Consolidating LLM (Phases 0–1)

Status: **draft, awaiting approval. No code written yet.**

## 0. Environment findings (checked, not assumed)
| Item | Found | Consequence |
|---|---|---|
| GPU | RTX 3050 Laptop, 4096 MiB, driver 566.07 | 4 GB budget governs everything |
| Python | only **3.12.10** installed; no 3.11 | See Decision D1 |
| WSL2 | **not installed** | Fallback for bitsandbytes costs a reboot-level install; avoid unless needed |
| git | 2.56 present, folder is not a repo | `git init` first |

## 1. Decisions I need from you
- **D1 – Python 3.11 vs 3.12.** Brief says 3.11. Options: (a) install 3.11 via `winget install Python.Python.3.11` (recommended, matches brief, ~1 min); (b) use 3.12 and pin versions accordingly. I'll do (a) unless you say otherwise.
- **D2 – bitsandbytes on Windows.** Recent bitsandbytes (>=0.43) ships Windows CUDA wheels. Plan: try it in Phase 0. If 4-bit load fails, I stop and report; I will *not* silently install WSL2 or drop to fp16 without telling you. Fallbacks in order: WSL2 (needs your OK, installation requires admin/reboot) → report.
- **D3 – Old-skill probe sources.** ARC-Easy (multiple-choice, loglik scoring), HellaSwag (loglik), GSM8K (generation, exact match; 0.5B will score low, so GSM8K-lite is a noisy signal). Proposal: 80 ARC-Easy + 80 HellaSwag + 40 GSM8K = 200 items, fixed seed, saved to `data/old_skill_probe.jsonl`, plus WikiText-2 perplexity on a fixed ~100k-token slice. OK?

## 2. Stack & versions
- venv `.venv`, `requirements.txt` pinned from what actually installs in Phase 0 (`pip freeze` of the working set, not guessed numbers).
- torch (CUDA 12.x wheel), transformers, peft, bitsandbytes, datasets, accelerate, pyyaml, pytest, sentence-transformers (for embedding dispersion; MiniLM-L6, ~90 MB, run on CPU to save VRAM), sacrebleu or own self-BLEU.
- Model: `Qwen/Qwen2.5-0.5B-Instruct` (dev), `Qwen/Qwen2.5-1.5B-Instruct` (final, not touched until after Phase 1).
- QLoRA: NF4, double-quant, bf16/fp16 compute (RTX 30-series supports bf16), r=16, alpha=32, dropout 0.05, all linear proj targets, gradient checkpointing, micro-batch 2–4 + grad accumulation, max seq len 256–384 to start. Shrink seq/batch before anything else if OOM. Never full fine-tune.
- Seeding: python/numpy/torch/cuda + `PYTHONHASHSEED`; seed stored in every result file.
- Config: `configs/*.yaml` loaded into frozen dataclasses (`config.py`), CLI override `--set a.b=c`. Logging: CSV for metrics, JSONL for samples/events; wandb off by default.

## 3. Repo layout (as specified, plus)
```
src/sleepllm/{config.py, utils.py, notebook.py, day.py, dream.py, diversity.py,
              replay.py (stub), sleep.py, eval.py, data/synthetic_world.py}
scripts/{phase0_checks.py, run_day.py, run_night.py, run_experiment.py (stub in P1), run_phase1.py}
configs/{dev_0p5b.yaml, final_1p5b.yaml}
tests/  results/  LOG.md  PLAN.md  requirements.txt
```
`run_phase1.py` is a thin orchestrator calling day → night → eval so Phase 1 is one reproducible command.

## 4. Component design

### synthetic_world.py
- Seeded generator of fictional entities: people, companies, cities, products, with attributes (role, employer, birthplace, founding year, HQ, product price, etc.). Names built from syllable generators so the base model has no prior (also verify: ask the base model a sample of questions and confirm ~chance accuracy → saved as a baseline number).
- Per day: 20–50 new facts as structured triples `(subject, relation, object)`; a configurable fraction (~20% from day 2 onward) are **updates** to earlier facts (e.g. CEO changed), which makes the "latest value wins" check meaningful.
- Each day also yields: source documents (templated prose, several surface forms per fact, with some distractor sentences), canonical QA, and **held-out QA phrasings**: question templates split into `train_templates` (usable by dreams) and `heldout_templates` (never shown to dream generator or trainer). Held-out templates are enforced by a test.
- Answers are exact strings, scored by normalised exact-match / containment.

### notebook.py
- JSONL append-only store `{day, id, text, subject, relation, source_doc, ts}`; read, filter by day, simple retrieval (BM25-style/keyword overlap, no extra dependency) → top-k notes formatted into the prompt. Update handling: later note on same (subject, relation) supersedes in retrieval.

### day.py
- For each document the base/current model, prompted (few-shot), extracts atomic fact notes → notebook. Extraction quality by a 0.5B model will be imperfect. I'll **measure it** (note-level recall vs the world's ground-truth facts) and report it. Fallback if too poor: rule-based/oracle extraction as an explicit, labelled ablation ("oracle notes"), never silently.

### dream.py (naive, Phase 1)
- From notes, the model generates: (a) QA pairs, (b) paraphrases, (c) counterfactual variations (clearly marked, trained as "if X were Y" so they don't overwrite the true fact — design detail to settle in Phase 2; in Phase 1 I generate them and **report their effect separately** via ablation if time allows).
- Plain sampling (temperature ~0.8), N samples per note. Dream QA is validated against the note (answer must appear in the note) — filtering stats logged. Only train-template phrasings are allowed.

### diversity.py
- distinct-1/2/3, self-BLEU (sampled pairs, capped), embedding dispersion (mean pairwise cosine distance via MiniLM). Pure functions on `list[str]`, unit-tested on hand-built cases (identical set → distinct≈low, self-BLEU≈1, dispersion≈0).
- Regulariser: **not implemented in Phase 1.**

### replay.py
- Stub: interface `ReplayBuffer.sample(n)` returning `[]`. Phase 1 trains on dreams only (as the brief says "no regulariser yet"; replay is Phase 2). I'll mention that the old-skill delta in Phase 1 therefore measures *unprotected* forgetting.

### sleep.py
- QLoRA SFT on dream samples, chat-template formatting, loss on answer tokens only. Epochs/LR from config (start: 3 epochs, lr 2e-4, cosine). Adapter saved per night. Merge-vs-stack: Phase 1 implements **stack-then-merge into a fresh adapter lineage is out of scope**; Phase 1 saves the adapter and evaluates with it loaded (no merge). Merge into a 4-bit base is lossy, so merging will be done to an fp16 copy only if needed later.

### eval.py
- New-knowledge QA (no notebook): held-out phrasings; also report on train phrasings separately. Conditions: **before night** (base, no notebook), **after night** (adapter, no notebook), **with notebook** (base + retrieved notes — upper bound). Updated facts scored separately.
- Old-skill: the 200-item probe + WikiText perplexity, before/after night.
- Greedy decoding, fixed seed, raw per-item outputs saved to `results/`.

## 5. Phase 0 (concrete checks)
1. Install 3.11 (D1), create `.venv`, install torch+deps.
2. Print `torch.cuda.is_available()`, GPU name, VRAM.
3. Load 0.5B in 4-bit, generate once, report peak VRAM (`torch.cuda.max_memory_allocated` and reserved).
4. 20-step QLoRA run on dummy data; report peak VRAM + wall time.
Output → `results/phase0.json`, summary in `LOG.md`. If bitsandbytes fails: stop, report, ask (D2).

## 6. Phase 1 run & report
`run_phase1.py` (seed fixed) → `results/phase1/` containing: world day-1 files, notes, dreams, adapter, eval raw outputs, `metrics.json`, `night_timing.json`.
Summary table I will give you (all numbers from the run):

| Metric | Value |
|---|---|
| Note extraction recall | |
| QA acc before night (no notebook), held-out / train phrasings | |
| QA acc after night (no notebook), held-out / train phrasings | |
| QA acc with notebook (upper bound) | |
| Old-skill probe before → after (per sub-task) | |
| WikiText ppl before → after | |
| Dream diversity: distinct-1/2/3, self-BLEU, dispersion | |
| Dream filter pass-rate, #samples | |
| Night wall time, peak VRAM | |

Caveats I will state with it: single seed, n=1 day, 0.5B model, so differences are indicative, not statistically established; I'll give binomial CIs for accuracies.

## 7. Tests (pytest, Phase 1)
- notebook: append/read roundtrip, supersede-on-update, retrieval ranking, persistence.
- synthetic_world: determinism by seed, fact counts in range, updates reference earlier facts, held-out templates disjoint from train templates, answers derivable from the facts.
- diversity: known-answer cases for distinct-n, self-BLEU, dispersion (with a fake embedder to avoid model download in tests).

## 8. Process rules
- `git init`; small commits (scaffold → config → world+tests → notebook+tests → diversity+tests → day → dream → sleep → eval → phase scripts → results). `.gitignore` for `.venv`, `results/**/adapter*`, HF cache. Commits end with the Co-Authored-By trailer.
- LOG.md appended after each step: command, result, failures.
- Stop after Phase 1. No regulariser, no replay content, no 30-day run.

## 9. Risks
| Risk | Mitigation |
|---|---|
| 0.5B model gives poor note extraction / dream QA | measure; oracle-notes ablation clearly labelled |
| 3 epochs on ~30 facts overfits to dream phrasing, held-out acc stays low | that is a legitimate finding; report it, don't tune on held-out set |
| bitsandbytes Windows issues | D2 above |
| Windows dataloader/multiprocessing quirks | `num_workers=0` |
| GSM8K probe too noisy at 0.5B | report per-task, don't aggregate blindly |
| Base model already knows names | syllable-generated names + measured baseline |

## 10. Time estimate
Phase 0: ~30 min incl. downloads. Phase 1 code+tests: a few hours of agent work; the night itself should be minutes on a 0.5B model (will be measured, not guessed).

**Awaiting your OK (and answers to D1–D3, or I proceed with the stated defaults).**
D1 resolved by user: use Python 3.12 (no 3.11 install).
