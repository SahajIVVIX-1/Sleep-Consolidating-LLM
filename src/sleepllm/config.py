"""YAML + dataclass config."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
import yaml


@dataclass
class ModelCfg:
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    load_4bit: bool = True
    max_seq_len: int = 384


@dataclass
class LoraCfg:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    epochs: int = 3
    lr: float = 2e-4
    micro_batch: int = 4
    grad_accum: int = 4
    gradient_checkpointing: bool = True


@dataclass
class WorldCfg:
    seed: int = 0
    facts_min: int = 20
    facts_max: int = 50
    update_frac: float = 0.2


@dataclass
class DreamCfg:
    samples_per_note: int = 6
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 64
    kinds: tuple = ("qa", "paraphrase", "counterfactual")


@dataclass
class EvalCfg:
    probe_arc: int = 80
    probe_hellaswag: int = 80
    probe_gsm8k: int = 40
    wikitext_tokens: int = 50000
    top_k_notes: int = 5


@dataclass
class Config:
    seed: int = 0
    out_dir: str = "results/phase1"
    model: ModelCfg = field(default_factory=ModelCfg)
    lora: LoraCfg = field(default_factory=LoraCfg)
    world: WorldCfg = field(default_factory=WorldCfg)
    dream: DreamCfg = field(default_factory=DreamCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)


def _merge(obj, d: dict):
    for k, v in d.items():
        cur = getattr(obj, k)
        if is_dataclass(cur):
            _merge(cur, v)
        else:
            if isinstance(cur, tuple):
                v = tuple(v)
            setattr(obj, k, v)
    return obj


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    cfg = Config()
    if path:
        _merge(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    for o in overrides or []:
        key, val = o.split("=", 1)
        parts = key.split(".")
        tgt = cfg
        for p in parts[:-1]:
            tgt = getattr(tgt, p)
        setattr(tgt, parts[-1], yaml.safe_load(val))
    return cfg


def to_dict(cfg: Config) -> dict:
    return asdict(cfg)
