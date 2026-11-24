"""Model loading and batched chat generation."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(name: str, load_4bit: bool = True):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    kw = {"device_map": {"": 0}, "dtype": torch.bfloat16, "low_cpu_mem_usage": True}
    if load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    model.eval()
    return tok, model


def chat_text(tok, user: str, system: str | None = None, assistant: str | None = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return t + assistant if assistant is not None else t


@torch.no_grad()
def generate(model, tok, users: list[str], max_new_tokens: int = 48, temperature: float = 0.0,
             top_p: float = 0.95, n: int = 1, batch_size: int = 16, system: str | None = None) -> list[list[str]]:
    """Returns, for each prompt, a list of n completions."""
    texts = [chat_text(tok, u, system) for u in users]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out: list[list[str]] = [[] for _ in texts]
    sample = temperature > 0
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        if sample:
            kw.update(do_sample=True, temperature=temperature, top_p=top_p, num_return_sequences=n)
        else:
            kw.update(do_sample=False)
        gen = model.generate(**enc, **kw)
        new = gen[:, enc["input_ids"].shape[1]:]
        dec = tok.batch_decode(new, skip_special_tokens=True)
        k = n if sample else 1
        for j, i in enumerate(idx):
            out[i] = [d.strip() for d in dec[j * k:(j + 1) * k]]
    return out
