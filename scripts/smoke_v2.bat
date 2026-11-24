@echo off
cd /d D:\03_Projects\Sleep-Consolidating-LLM
.venv\Scripts\python.exe scripts\run_pilot.py --days 2 --arms both_v2 --set out_dir=results/smoke2 dream.samples_per_note=1 lora.epochs=1 eval.wikitext_tokens=2000 > results\smoke2\run.log 2>&1
echo SMOKE DONE >> results\smoke2\run.log
