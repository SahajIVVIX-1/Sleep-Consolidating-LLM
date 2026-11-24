# Crash evidence (2026-11-24)
- 20:09:15 python.exe access violation 0xc0000005 in python312.dll (Windows Application Error 1000 / WER 1001), 11 s later an
  `nvlddmkm` (NVIDIA display driver) event 153. Fault handler stack (`faulthandler_both_v2_crash_2009.log`): inside
  `bitsandbytes/backends/cuda/ops.py::_gemm_4bit_kernel_impl` (fused 4-bit GEMM, `cgemm_4bit_*`), called from a Qwen2 forward
  during dream generation of both_v2 day 1, ~10 min into the night.
- Earlier silent deaths of replay_v2 (night 3, twice) and the visible debug run left no trace; they are consistent with the
  same native crash but that was NOT proven (no fault handler was enabled then).
- Mitigation: sleepllm.llm.disable_fused_4bit_gemm() (bnb falls back to dequantise + F.linear). Mean |logit diff| fused vs fallback 0.13 on logits of mean |.| 2.8.
