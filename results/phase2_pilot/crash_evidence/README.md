# Crash evidence (2026-11-24) -- corrected diagnosis
Two crashes of the both_v2 arm (20:09:15 and 20:22:43), both exactly 10:01 after process start:
- Windows Application Error 1000 / WER 1001: python.exe, faulting module python312.dll, exception 0xc0000005,
  **identical fault offset 0x27e990 both times**. An `nvlddmkm` event 153 follows each crash by 8-11 s (consequence of the
  process dying, not a cause).
- Each faulthandler log starts with "Timeout (0:10:00)!" followed by "Windows fatal exception: access violation".
  That header is the output of `faulthandler.dump_traceback_later(600, repeat=True)`, a debugging aid I had added
  to scripts/run_pilot.py. The thread being dumped was in `_gemm_4bit_kernel_impl` the first time and in
  `Linear4bit.forward` the second time, i.e. just wherever bitsandbytes happened to be running.
- Supporting: the surviving debug run had 3 such timer dumps without a crash, all during training/eval (not during
  generation); both crashes had the dump land during dream generation (adaptive both_v2 generation lasts > 10 min).
  reg_v2 (same long adaptive generation, no timer) ran fine.
- My first diagnosis ("fused bitsandbytes 4-bit GEMM kernel") was WRONG; the fused-kernel workaround is now opt-in only
  (SLEEPLLM_DISABLE_FUSED_4BIT_GEMM=1) and was not used for any reported result.
- NOT explained: the three earlier silent deaths (replay_v2 night 3 twice, visible debug run at 19:16) happened before / without
  the timer in play; the visible-window hypothesis is unproven. Hidden-window launches have been used since.
