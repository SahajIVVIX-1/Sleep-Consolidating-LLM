"""Phase 2 pilot: N simulated days, 4 arms (naive | reg | replay | both), same seed and same world."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import torch

from . import dream as dream_mod
from . import eval as ev
from . import sleep as sleep_mod
from .config import Config, to_dict
from .data.synthetic_world import World, day_to_dict
from .day import run_day
from .llm import load_model
from .notebook import Notebook
from .replay import ReplayBuffer, assert_disjoint, load_old_skill
from .utils import append_csv, seed_everything, write_json, write_jsonl

ARMS = {"naive": (False, False), "reg": (True, False), "replay": (False, True), "both": (True, True)}


def _free():
    gc.collect()
    torch.cuda.empty_cache()


def shared_stage(cfg: Config, days: int, out: Path):
    """World, base-model note extraction (shared by all arms), day-0 baselines."""
    sh = out / "shared"
    sh.mkdir(parents=True, exist_ok=True)
    world = World(cfg.world.seed, cfg.world.facts_min, cfg.world.facts_max, cfg.world.update_frac)
    dds = [world.generate_day(d) for d in range(1, days + 1)]
    for dd in dds:
        write_json(sh / f"world_day{dd.day}.json", day_to_dict(dd))
    probe = ev.load_probe(cfg.eval)
    base_file = sh / "baseline.json"
    nb_path = sh / "notebook_all.jsonl"
    if base_file.exists() and nb_path.exists():
        return dds, probe, Notebook(nb_path)
    seed_everything(cfg.seed)
    tok, model = load_model(cfg.model.name, cfg.model.load_4bit)
    base = {}
    base["old_skill"], _ = ev.eval_old_skills(model, tok, probe)
    base["wikitext_ppl"] = ev.wikitext_ppl(model, tok, cfg.eval.wikitext_tokens)
    nb = Notebook(nb_path)
    ext = []
    for dd in dds:
        st = run_day(model, tok, nb, dd.day, dd.documents, dd.facts)
        st.pop("raw_examples", None)
        ext.append({"day": dd.day, **st})
    base["extraction"] = ext
    write_json(base_file, base)
    del model
    _free()
    return dds, probe, nb


def _qa_acc(model, tok, dd, today_ids: set):
    held = [q for q in dd.qa if q.split == "heldout"]
    summ, rows = ev.eval_new_knowledge(model, tok, held, None)
    today = [r for r in rows if r["fact_id"] in today_ids]
    earlier = [r for r in rows if r["fact_id"] not in today_ids]
    return {"all": ev._acc(rows), "today": ev._acc(today), "earlier": ev._acc(earlier) if earlier else None}, rows


def run_arm(cfg: Config, arm: str, days: int, dds, probe, nb_all: Notebook, out: Path) -> None:
    reg, rep = ARMS[arm]
    cfg.dream.regulariser = reg
    cfg.replay.enabled = rep
    adir = out / arm
    adir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    tok, model = load_model(cfg.model.name, cfg.model.load_4bit)
    model = sleep_mod.attach_adapter(model, cfg.lora)
    state = dream_mod.DreamState()
    rb = None
    if rep:
        old = load_old_skill(cfg.replay.n_old_skill, cfg.seed)
        assert_disjoint(old, probe)
        rb = ReplayBuffer(old, cfg.replay.reservoir_cap, cfg.seed)
    csv = out / "metrics.csv"
    for dd in dds[:days]:
        d = dd.day
        view = nb_all.upto(d)
        notes = [n for n in view.active() if n.day == d]
        torch.cuda.reset_peak_memory_stats()
        t_night = time.time()
        samples, ds = dream_mod.make_dreams(model, tok, notes, cfg.dream, state, d, seed=cfg.seed + d)
        n_dream = len(samples)
        replay = rb.sample(n_dream, cfg.replay.ratio) if rb else []
        write_jsonl(adir / f"dreams_day{d}.jsonl", samples)
        ts = sleep_mod.train(model, tok, samples + replay, cfg, cfg.seed + d,
                             log=lambda r, d=d: append_csv(adir / "train_log.csv", {"day": d, **r}))
        if rb:
            rb.add_dreams(samples)
        night_s = time.time() - t_night
        peak_night = torch.cuda.max_memory_allocated() / 2**20
        # ---- evaluation (adapter on, no notebook)
        today_ids = {f.id for f in dd.facts}
        qa, rows = _qa_acc(model, tok, dd, today_ids)
        write_jsonl(adir / f"qa_day{d}.jsonl", rows)
        old_s, _ = ev.eval_old_skills(model, tok, probe)
        ppl = ev.wikitext_ppl(model, tok, cfg.eval.wikitext_tokens)
        m = ds["metrics"]
        row = {
            "arm": arm, "day": d, "acc_today": qa["today"]["acc"], "n_today": qa["today"]["n"],
            "acc_earlier": qa["earlier"]["acc"] if qa["earlier"] else "", "n_earlier": qa["earlier"]["n"] if qa["earlier"] else 0,
            "acc_all": qa["all"]["acc"], "old_overall": old_s["overall"]["acc"], "old_arc": old_s["arc_easy"]["acc"],
            "old_hellaswag": old_s["hellaswag"]["acc"], "old_gsm8k": old_s["gsm8k"]["acc"], "wikitext_ppl": ppl,
            "distinct_1": m["distinct_1"], "distinct_2": m["distinct_2"], "distinct_3": m["distinct_3"],
            "self_bleu": m["self_bleu"], "embedding_dispersion": m["embedding_dispersion"],
            "vocab_entropy": m["vocab_entropy"], "entropy_ratio": m["entropy_ratio"],
            "n_dreams": n_dream, "n_replay": len(replay), "n_candidates": ds["n_candidates"],
            "first_pass_yield": ds["first_pass_yield"], "final_fill": ds["final_fill"],
            "facts_covered": ds["facts_covered"], "n_notes": ds["n_notes"],
            "alarm": "+".join(ds["alarm"]), "temp_bump": state.temp_bump, "variety_bump": state.variety_bump,
            "dream_s": ds["seconds"], "train_s": ts["train_seconds"], "night_s": night_s,
            "peak_vram_night_mb": peak_night, "peak_vram_total_mb": torch.cuda.max_memory_allocated() / 2**20,
            "loss_last": ts["loss_last"],
        }
        append_csv(csv, row)
        write_json(adir / f"day{d}.json", {"row": row, "dream_stats": ds, "train": ts, "qa": qa, "old_skill": old_s,
                                           "alarms": state.alarms})
        print(arm, d, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                       if k in ("acc_today", "acc_earlier", "old_overall", "wikitext_ppl", "distinct_2", "night_s")},
              flush=True)
    write_json(adir / "done.json", {"arm": arm, "days": days})
    del model
    _free()


def run_pilot(cfg: Config, days: int = 5, arms: list[str] | None = None) -> None:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", to_dict(cfg))
    dds, probe, nb = shared_stage(cfg, days, out)
    for arm in arms or list(ARMS):
        if (out / arm / "done.json").exists():
            print("skip finished arm", arm, flush=True)
            continue
        csv = out / "metrics.csv"
        if csv.exists():  # drop rows of an unfinished earlier attempt of this arm
            lines = csv.read_text(encoding="utf-8").splitlines()
            keep = [lines[0]] + [l for l in lines[1:] if not l.startswith(arm + ",")]
            csv.write_text(chr(10).join(keep) + chr(10), encoding="utf-8")
        run_arm(cfg, arm, days, dds, probe, nb, out)
