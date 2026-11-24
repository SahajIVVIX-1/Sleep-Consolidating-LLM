@echo off
cd /d D:\03_Projects\Sleep-Consolidating-LLM
.venv\Scripts\python.exe scripts\run_pilot.py --days 5 --arms both_v2 --set out_dir=results/phase3/sweep_S1 lora.lr=0.0002 > results\phase3\sweep_S1\run.log 2>&1
if errorlevel 1 (echo S1 FAILED >> results\phase3\sweep_S1\run.log & exit /b 1)
.venv\Scripts\python.exe scripts\run_pilot.py --days 5 --arms both_v2 --set out_dir=results/phase3/sweep_S2 replay.past_frac=0.25 > results\phase3\sweep_S2\run.log 2>&1
if errorlevel 1 (echo S2 FAILED >> results\phase3\sweep_S2\run.log & exit /b 1)
.venv\Scripts\python.exe scripts\run_pilot.py --days 5 --arms both_v2 --set out_dir=results/phase3/sweep_S3 lora.epochs=2 lora.replay_epochs=1 > results\phase3\sweep_S3\run.log 2>&1
if errorlevel 1 (echo S3 FAILED >> results\phase3\sweep_S3\run.log & exit /b 1)
echo SWEEP DONE >> results\phase3\sweep_S3\run.log
