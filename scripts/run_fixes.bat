@echo off
cd /d D:\03_Projects\Sleep-Consolidating-LLM
.venv\Scripts\python.exe scripts\run_pilot.py --days 5 --arms both_v2 --set out_dir=results/phase3/fix_P1 lora.lr=0.0002 replay.text_frac=0.12 > results\phase3\fix_P1\run.log 2>&1
if errorlevel 1 (echo P1 FAILED >> results\phase3\fix_P1\run.log & exit /b 1)
.venv\Scripts\python.exe scripts\run_pilot.py --days 5 --arms both_v2 --set out_dir=results/phase3/fix_P2 lora.lr=0.0002 lora.lr_min=0.00005 > results\phase3\fix_P2\run.log 2>&1
if errorlevel 1 (echo P2 FAILED >> results\phase3\fix_P2\run.log & exit /b 1)
.venv\Scripts\python.exe scripts\run_pilot.py --days 5 --arms both_v2 --set out_dir=results/phase3/fix_P3 lora.lr=0.0002 lora.kl_weight=0.1 > results\phase3\fix_P3\run.log 2>&1
if errorlevel 1 (echo P3 FAILED >> results\phase3\fix_P3\run.log & exit /b 1)
echo FIXES DONE >> results\phase3\fix_P3\run.log
