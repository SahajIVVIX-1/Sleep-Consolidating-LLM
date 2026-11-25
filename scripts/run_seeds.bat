@echo off
cd /d D:\03_Projects\Sleep-Consolidating-LLM
.venv\Scripts\python.exe scripts\run_pilot.py --days 5 --arms naive --set out_dir=results/phase3/seed1 seed=1 world.seed=1 > results\phase3\seed1\run_naive.log 2>&1
if errorlevel 1 (echo seed1 naive FAILED >> results\phase3\seed1\run_naive.log & exit /b 1)
.venv\Scripts\python.exe scripts\run_pilot.py --days 5 --arms both_v2 --set out_dir=results/phase3/seed1 seed=1 world.seed=1 lora.lr=0.0002 > results\phase3\seed1\run_both.log 2>&1
if errorlevel 1 (echo seed1 both FAILED >> results\phase3\seed1\run_both.log & exit /b 1)
.venv\Scripts\python.exe scripts\run_pilot.py --days 5 --arms naive --set out_dir=results/phase3/seed2 seed=2 world.seed=2 > results\phase3\seed2\run_naive.log 2>&1
if errorlevel 1 (echo seed2 naive FAILED >> results\phase3\seed2\run_naive.log & exit /b 1)
.venv\Scripts\python.exe scripts\run_pilot.py --days 5 --arms both_v2 --set out_dir=results/phase3/seed2 seed=2 world.seed=2 lora.lr=0.0002 > results\phase3\seed2\run_both.log 2>&1
if errorlevel 1 (echo seed2 both FAILED >> results\phase3\seed2\run_both.log & exit /b 1)
echo SEEDS DONE >> results\phase3\seed2\run_both.log
