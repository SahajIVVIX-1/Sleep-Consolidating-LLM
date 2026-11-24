@echo off
cd /d D:\03_Projects\Sleep-Consolidating-LLM
.venv\Scripts\python.exe scripts\run_pilot.py --days 5 --arms reg_v2 > results\phase2_pilot\run_reg_v2.log 2>&1
if errorlevel 1 (echo reg_v2 FAILED & exit /b 1)
.venv\Scripts\python.exe scripts\run_pilot.py --days 5 --arms replay_v2 > results\phase2_pilot\run_replay_v2.log 2>&1
if errorlevel 1 (echo replay_v2 FAILED & exit /b 1)
.venv\Scripts\python.exe scripts\run_pilot.py --days 5 --arms both_v2 > results\phase2_pilot\run_both_v2.log 2>&1
echo ALL V2 DONE
