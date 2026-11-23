"""One day/night cycle end to end (Phase 1). Used by scripts/run_phase1.py, run_day.py, run_night.py."""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import torch

from . import day as day_mod
from . import dream as dream_mod
from . import eval as ev
from . import sleep as sleep_mod
from .config import Config, to_dict
from .data.synthetic_world import QA, World, day_to_dict
from .diversity import all_metrics
from .llm import load_model
from .notebook import Notebook
from .replay import ReplayBuffer
from .utils import append_csv, seed_everything, write_json, write_jsonl


def make_day(cfg: Config, day: int = 1):
    w = World(cfg.world.seed, cfg.world.facts_min, cfg.world.facts_max, cfg.world.update_frac)
    d = None
    for k in range(1, day + 1):
        d = w.generate_day(k)
    return w, d


def day_stage(cfg: Config, model, tok, day_data, out: Path) -> dict:
    nb_path = out / "notebook.jsonl"
    if nb_path.exists():
        nb_path.unlink()
    nb = Notebook(nb_path)
    t0 = time.time()
    stats = day_mod.run_day(model, tok, nb, day_data.day, day_data.documents, day_data.facts)
    stats["seconds"] = time.time() - t0
    return stats, nb


def night_dream_stage(cfg: Config, model, tok, nb: Notebook, out: Path):
    t0 = time.time()
    samples, stats = dream_mod.dream(model, tok, nb, cfg.dream)
    stats["dream_seconds"] = time.time() - t0
    write_jsonl(out / "dreams.jsonl", samples)
    return samples, stats


def dream_diversity(samples: list[dict]) -> dict:
    texts = [f"{s['user']} {s['assistant']}" for s in samples]
    res = {"all": all_metrics(texts)}
    for k in sorted({s["kind"] for s in samples}):
        res[k] = all_metrics([f"{s['user']} {s['assistant']}" for s in samples if s["kind"] == k])
    res["exact_duplicate_frac"] = 1 - len(set(texts)) / max(len(texts), 1)
    return res


def night_train_stage(cfg: Config, model, tok, samples: list[dict], out: Path):
    replay = ReplayBuffer().sample(0)  # stub in Phase 1: no replay
    train_samples = samples + replay
    model = sleep_mod.attach_adapter(model, cfg.lora)
    log_path = out / "train_log.csv"
    if log_path.exists():
        log_path.unlink()
    stats = sleep_mod.train(model, tok, train_samples, cfg, cfg.seed, log=lambda r: append_csv(log_path, r))
    model.save_pretrained(out / "adapter")
    return model, stats


def run_phase1(cfg: Config) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    write_json(out / "config.json", to_dict(cfg))
    world, dd = make_day(cfg, 1)
    write_json(out / "day1_world.json", day_to_dict(dd))
    qa: list[QA] = dd.qa
    tok, model = load_model(cfg.model.name, cfg.model.load_4bit)
    probe = ev.load_probe(cfg.eval)
    M: dict = {"model": cfg.model.name, "seed": cfg.seed, "n_facts": len(dd.facts), "n_qa": len(qa)}

    # ---- before night (base model)
    M["qa_before_no_notebook"], rows = ev.eval_new_knowledge(model, tok, qa, None)
    write_jsonl(out / "qa_before_no_notebook.jsonl", rows)
    M["old_skill_before"], rows = ev.eval_old_skills(model, tok, probe)
    M["wikitext_ppl_before"] = ev.wikitext_ppl(model, tok, cfg.eval.wikitext_tokens)

    # ---- day
    M["day"], nb = day_stage(cfg, model, tok, dd, out)
    M["qa_with_notebook_base"], rows = ev.eval_new_knowledge(model, tok, qa, nb, cfg.eval.top_k_notes)
    write_jsonl(out / "qa_with_notebook_base.jsonl", rows)

    # ---- night
    t_night = time.time()
    samples, M["dream"] = night_dream_stage(cfg, model, tok, nb, out)
    t_dream_done = time.time()
    model, M["train"] = night_train_stage(cfg, model, tok, samples, out)
    M["night_seconds"] = time.time() - t_night
    M["night_seconds_dream"] = t_dream_done - t_night
    M["night_seconds_train"] = M["train"]["train_seconds"]
    M["dream_diversity"] = dream_diversity(samples)  # computed outside the timed night

    # ---- after night
    M["qa_after_no_notebook"], rows = ev.eval_new_knowledge(model, tok, qa, None)
    write_jsonl(out / "qa_after_no_notebook.jsonl", rows)
    M["qa_after_with_notebook"], rows = ev.eval_new_knowledge(model, tok, qa, nb, cfg.eval.top_k_notes)
    M["old_skill_after"], rows = ev.eval_old_skills(model, tok, probe)
    M["wikitext_ppl_after"] = ev.wikitext_ppl(model, tok, cfg.eval.wikitext_tokens)
    M["peak_vram_total_mb"] = torch.cuda.max_memory_allocated() / 2**20
    write_json(out / "metrics.json", M)
    return M
