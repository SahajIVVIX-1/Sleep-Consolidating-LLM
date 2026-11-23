"""Phase 0: CUDA check, 4-bit load + generate, 20-step QLoRA on dummy data. Writes results/phase0.json."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from sleepllm.utils import seed_everything, write_json  # noqa: E402


def mb(x):
    return round(x / 2**20, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="results/phase0.json")
    ap.add_argument("--seq", type=int, default=384)
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    seed_everything(0)
    res = {"model": a.model}

    res["cuda_available"] = torch.cuda.is_available()
    props = torch.cuda.get_device_properties(0)
    res["gpu"] = props.name
    res["vram_total_mb"] = mb(props.total_memory)
    res["torch"] = torch.__version__
    print(res, flush=True)

    import bitsandbytes
    import peft
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    res["versions"] = {"transformers": transformers.__version__, "peft": peft.__version__,
                       "bitsandbytes": bitsandbytes.__version__}

    tok = AutoTokenizer.from_pretrained(a.model)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, device_map={"": 0},
                                                 torch_dtype=torch.bfloat16)
    res["load_seconds"] = round(time.time() - t0, 1)
    res["vram_after_load_mb"] = mb(torch.cuda.memory_allocated())

    msgs = [{"role": "user", "content": "Say hello in one sentence."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
    t0 = time.time()
    out = model.generate(**ids, max_new_tokens=32, do_sample=False)
    res["generate_seconds"] = round(time.time() - t0, 2)
    res["generation"] = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    res["peak_vram_generate_mb"] = mb(torch.cuda.max_memory_allocated())
    res["peak_vram_generate_reserved_mb"] = mb(torch.cuda.max_memory_reserved())
    print(res, flush=True)

    # 20-step QLoRA on dummy data
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))
    res["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    losses = []
    for step in range(20):
        x = torch.randint(0, tok.vocab_size, (a.batch, a.seq), device="cuda")
        loss = model(input_ids=x, labels=x).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(round(loss.item(), 3))
    torch.cuda.synchronize()
    res["train20_seconds"] = round(time.time() - t0, 1)
    res["train_batch"] = a.batch
    res["train_seq"] = a.seq
    res["train_losses_first_last"] = [losses[0], losses[-1]]
    res["peak_vram_train_mb"] = mb(torch.cuda.max_memory_allocated())
    res["peak_vram_train_reserved_mb"] = mb(torch.cuda.max_memory_reserved())
    write_json(a.out, res)
    print(res)


if __name__ == "__main__":
    main()
