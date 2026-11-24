@echo off
cd /d D:\03_Projects\Sleep-Consolidating-LLM
.venv\Scripts\python.exe scripts\profile_replay_night.py 0 results\phase3\profile_full_logits.csv 1 > results\phase3\profile_full_logits.log 2>&1
.venv\Scripts\python.exe scripts\profile_replay_night.py 1 results\phase3\profile_lean_logits.csv 1 > results\phase3\profile_lean_logits.log 2>&1
echo PROFILES DONE >> results\phase3\profile_lean_logits.log
