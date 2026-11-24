@echo off
cd /d D:\03_Projects\Sleep-Consolidating-LLM
.venv\Scripts\python.exe scripts\run_pilot.py --days 5 --arms replay > results\phase2_pilot\run_replay.log 2>&1
if errorlevel 1 (echo replay failed, see log & exit /b 1)
.venv\Scripts\python.exe scripts\run_pilot.py --days 5 --arms both > results\phase2_pilot\run_both.log 2>&1
echo ALL DONE
