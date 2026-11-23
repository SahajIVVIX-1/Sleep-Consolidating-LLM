"""Night, step 2: QLoRA training of an adapter on dreams (+ replay, stub in Phase 1)."""
from __future__ import annotations

import math
import random
import time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .llm import chat_text

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def attach_adapter(model, lora_cfg):
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=lora_cfg.gradient_checkpointing)
    return get_peft_model(model, LoraConfig(r=lora_cfg.r, lora_alpha=lora_cfg.alpha, lora_dropout=lora_cfg.dropout,
                                            task_type="CAUSAL_LM", target_modules=TARGETS))


def _encode(tok, sample, max_len):
    prompt = chat_text(tok, sample["user"])
    full = prompt + sample["assistant"] + "<|im_end|>"
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    f = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * min(len(p), len(f)) + f[len(p):]
    return f, labels


def _batch(tok, items):
    L = max(len(i[0]) for i in items)
    ids = torch.full((len(items), L), tok.pad_token_id)
    lab = torch.full((len(items), L), -100)
    att = torch.zeros((len(items), L), dtype=torch.long)
    for k, (f, l) in enumerate(items):
        ids[k, :len(f)] = torch.tensor(f)
        lab[k, :len(l)] = torch.tensor(l)
        att[k, :len(f)] = 1
    return ids.cuda(), lab.cuda(), att.cuda()


def train(model, tok, samples: list[dict], cfg, seed: int = 0, log=None) -> dict:
    """Train the (already attached) LoRA adapter. Only assistant tokens carry loss."""
    c = cfg.lora
    rng = random.Random(seed)
    enc = [_encode(tok, s, cfg.model.max_seq_len) for s in samples]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=c.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(enc) / (c.micro_batch * c.grad_accum))
    total = steps_per_epoch * c.epochs
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    model.train()
    step, losses = 0, []
    for ep in range(c.epochs):
        order = list(range(len(enc)))
        rng.shuffle(order)
        mbs = [order[i:i + c.micro_batch] for i in range(0, len(order), c.micro_batch)]
        for g in range(0, len(mbs), c.grad_accum):
            group = mbs[g:g + c.grad_accum]
            ntok = sum(sum(1 for x in enc[i][1][1:] if x != -100) for mb in group for i in mb)
            tot = 0.0
            for mb in group:
                ids, lab, att = _batch(tok, [enc[i] for i in mb])
                logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1),
                    ignore_index=-100, reduction="sum") / max(ntok, 1)
                loss.backward()
                tot += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            losses.append(tot)
            if log:
                log({"step": step, "epoch": ep, "loss": tot, "lr": sched.get_last_lr()[0]})
    torch.cuda.synchronize()
    model.eval()
    return {"train_seconds": time.time() - t0, "steps": step, "n_samples": len(enc),
            "loss_first": losses[0], "loss_last": losses[-1],
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20}
