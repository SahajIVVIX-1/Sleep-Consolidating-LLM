"""Phase 2 pilot: N simulated days, 4 arms (naive | reg | replay | both), same seed and same world.

Run ONE arm per Python process (scripts/run_pilot.py --arms <arm>). Each finished day is checkpointed
(adapter + dream/replay state) so an interrupted arm resumes at the next day; rows are appended to metrics.csv.
"""
from __future__ import annotations

import csv
import gc
import json
import shutil
import threading
import time
from dataclasses import asdict
from pathlib import Path

import psutil
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

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


def free_ram_gb() -> float:
    return psutil.virtual_memory().available / 2**30


class ResourceLog:
    """Samples process RSS, free system RAM and VRAM every `every` seconds into a CSV."""

    def __init__(self, path: Path, every: float = 20.0):
        self.path, self.every = path, every
        self.proc = psutil.Process()
        self.peak_rss = 0.0
        self.min_free = 1e9
        self.peak_vram = 0.0
        self._stop = threading.Event()
        self.tag = ""

    def _sample(self):
        rss = self.proc.memory_info().rss / 2**20
        free = free_ram_gb()
        vram = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
        self.peak_rss, self.min_free, self.peak_vram = max(self.peak_rss, rss), min(self.min_free, free), max(self.peak_vram, vram)
        append_csv(self.path, {"t": time.strftime("%H:%M:%S"), "tag": self.tag, "rss_mb": round(rss),
                               "free_ram_gb": round(free, 2), "vram_peak_mb": round(vram)})

    def start(self):
        def loop():
            while not self._stop.wait(self.every):
                self._sample()
        self._sample()
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        self._sample()


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


# ---------------------------------------------------------------- checkpoint / resume
def _done_days(csv_path: Path, arm: str) -> list[int]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        return sorted(int(r["day"]) for r in csv.DictReader(f) if r["arm"] == arm)


def _save_ckpt(model, adir: Path, day: int, state, rb) -> None:
    for old in adir.glob("adapter_day*"):
        shutil.rmtree(old, ignore_errors=True)  # keep only the latest (and it is gitignored)
    model.save_pretrained(adir / f"adapter_day{day}")
    write_json(adir / "state.json", {
        "day": day, "dream_state": asdict(state),
        "reservoir": rb.reservoir if rb else [], "seen": rb.seen if rb else 0})


def _load_ckpt(model, adir: Path, day: int, state, rb) -> None:
    sd = load_file(str(adir / f"adapter_day{day}" / "adapter_model.safetensors"))
    set_peft_model_state_dict(model, sd)
    st = json.loads((adir / "state.json").read_text(encoding="utf-8"))
    assert st["day"] == day
    for k, v in st["dream_state"].items():
        setattr(state, k, v)
    if rb:
        rb.reservoir, rb.seen = st["reservoir"], st["seen"]


def run_arm(cfg: Config, arm: str, days: int, dds, probe, nb_all: Notebook, out: Path) -> None:
    reg, rep = ARMS[arm]
    cfg.dream.regulariser = reg
    cfg.replay.enabled = rep
    adir = out / arm
    adir.mkdir(parents=True, exist_ok=True)
    csv_path = out / "metrics.csv"
    done = _done_days(csv_path, arm)
    resume_from = done[-1] if done else 0
    if resume_from and not (adir / f"adapter_day{resume_from}").exists():
        raise SystemExit(f"{arm}: rows for days {done} exist but no adapter checkpoint; refusing to guess. "
                         f"Delete those rows (and the arm folder) to restart the arm.")
    seed_everything(cfg.seed)
    rlog = ResourceLog(out / f"resources_{arm}.csv")
    print(f"[{arm}] free RAM {free_ram_gb():.1f} GB, VRAM used "
          f"{torch.cuda.mem_get_info()[1] / 2**20 - torch.cuda.mem_get_info()[0] / 2**20:.0f} MB; resume after day {resume_from}",
          flush=True)
    rlog.start()
    tok, model = load_model(cfg.model.name, cfg.model.load_4bit)
    model = sleep_mod.attach_adapter(model, cfg.lora)
    state = dream_mod.DreamState()
    rb = None
    if rep:
        old = load_old_skill(cfg.replay.n_old_skill, cfg.seed)  # capped at n_old_skill items, cached on disk
        assert_disjoint(old, probe)
        rb = ReplayBuffer(old, cfg.replay.reservoir_cap, cfg.seed)
    if resume_from:
        _load_ckpt(model, adir, resume_from, state, rb)
        print(f"[{arm}] resumed from checkpoint of day {resume_from}", flush=True)
    for dd in dds[:days]:
        d = dd.day
        if d <= resume_from:
            continue
        rlog.tag = f"{arm} day{d}"
        for f in (adir / f"dreams_day{d}.jsonl", adir / f"qa_day{d}.jsonl"):
            f.unlink(missing_ok=True)
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
        _free()
        if rb:
            rb.add_dreams(samples)
        night_s = time.time() - t_night
        peak_night = torch.cuda.max_memory_allocated() / 2**20
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
        write_json(adir / f"day{d}.json", {"row": row, "dream_stats": ds, "train": ts, "qa": qa, "old_skill": old_s,
                                           "alarms": state.alarms})
        _save_ckpt(model, adir, d, state, rb)   # checkpoint first,
        append_csv(csv_path, row)              # then the row: a row implies a usable checkpoint
        _free()
        print(arm, d, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                       if k in ("acc_today", "acc_earlier", "old_overall", "wikitext_ppl", "distinct_2", "night_s")},
              f"| rss {rlog.proc.memory_info().rss / 2**20:.0f} MB, free RAM {free_ram_gb():.1f} GB", flush=True)
    rlog.stop()
    write_json(adir / "done.json", {"arm": arm, "days": days, "peak_rss_mb": rlog.peak_rss,
                                    "min_free_ram_gb": rlog.min_free, "peak_vram_mb": rlog.peak_vram})
    del model, tok
    _free()


def run_pilot(cfg: Config, days: int = 5, arms: list[str] | None = None) -> None:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfgp = out / "config.json"
    if not cfgp.exists():
        write_json(cfgp, to_dict(cfg))
    dds, probe, nb = shared_stage(cfg, days, out)
    for arm in arms or list(ARMS):
        if (out / arm / "done.json").exists():
            print("skip finished arm", arm, flush=True)
            continue
        run_arm(cfg, arm, days, dds, probe, nb, out)
