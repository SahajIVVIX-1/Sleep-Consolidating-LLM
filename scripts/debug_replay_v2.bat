@echo off
cd /d D:\03_Projects\Sleep-Consolidating-LLM
set PYTHONFAULTHANDLER=1
set CUDA_LAUNCH_BLOCKING=1
echo START %date% %time% > results\phase2_pilot\debug_replay_v2.log
.venv\Scripts\python -X faulthandler scripts/run_pilot.py --days 3 --arms replay_v2 >> results/phase2_pilot/debug_replay_v2.log 2>&1
echo PYTHON EXITCODE %errorlevel% at %date% %time% >> results\phase2_pilot\debug_replay_v2.log
