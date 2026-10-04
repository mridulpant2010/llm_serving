"""Experiment configuration: YAML loading and expansion into server runs.

One *server run* = one vLLM launch with a fixed (KV dtype, speculation method,
prefix-caching) setting.  Inside a server run we sweep context length and
concurrency, because those can be changed without restarting the server.
Restarting is the expensive part (model load), so we keep it to a minimum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_KV_DTYPES = {"auto", "fp8", "fp8_e4m3", "fp8_e5m2", "float16", "bfloat16"}
VALID_WORKLOADS = {"synthetic", "corpus", "sharegpt", "shared_prefix"}


@dataclass
class FixedSettings:
    """Controls that are held constant across every run."""

    output_len: int = 256
    gpu_memory_utilization: float = 0.90
    seed: int = 0
    max_num_seqs: int = 256  # raised automatically if a stage sweeps higher
    repetitions: int = 3
    rounds_per_level: int = 4  # requests per level = concurrency * rounds
    min_requests_per_level: int = 16
    warmup_requests: int = 16
    request_timeout_s: float = 900.0
    server_start_timeout_s: float = 1200.0
    enforce_eager: bool = False
    stop_on_failure_fraction: float = 0.5  # skip higher C once this many fail
    extra_server_args: list[str] = field(default_factory=list)


@dataclass
class ServerRun:
    """One server launch plus the sweep that runs against it."""

    stage: str
    kv_dtype: str
    spec_name: str
    spec_config: dict[str, Any] | None
    prefix_caching: bool
    workload: str
    workload_args: dict[str, Any]
    context_lengths: tuple[int, ...]
    concurrency: tuple[int, ...]
    max_model_len: int
    max_num_seqs: int

    @property
    def run_id(self) -> str:
        return (
            f"{self.stage}__kv-{self.kv_dtype}__spec-{self.spec_name}"
            f"__pc-{int(self.prefix_caching)}"
        )


@dataclass
class ExperimentConfig:
    model: str
    tokenizer: str | None = None
    dtype: str = "auto"
    quantization: str | None = None
    fixed: FixedSettings = field(default_factory=FixedSettings)
    speculation: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    data: dict[str, str] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer or self.model


_STAGE_REQUIRED = ("kv_dtypes", "speculation", "context_lengths", "concurrency")


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "model" not in raw:
        raise ValueError(f"{path}: config must be a mapping with a 'model' key")

    try:
        fixed = FixedSettings(**(raw.get("fixed") or {}))
    except TypeError as e:  # unknown key -> most likely a typo
        raise ValueError(f"{path}: bad 'fixed' section: {e}") from e

    cfg = ExperimentConfig(
        model=raw["model"],
        tokenizer=raw.get("tokenizer"),
        dtype=raw.get("dtype", "auto"),
        quantization=raw.get("quantization"),
        fixed=fixed,
        speculation=raw.get("speculation") or {"none": None},
        stages=raw.get("stages") or {},
        data=raw.get("data") or {},
        quality=raw.get("quality") or {},
    )
    cfg.speculation.setdefault("none", None)
    validate(cfg)
    return cfg


def validate(cfg: ExperimentConfig) -> None:
    for name, stage in cfg.stages.items():
        missing = [k for k in _STAGE_REQUIRED if k not in stage]
        if missing:
            raise ValueError(f"stage '{name}' is missing keys: {missing}")
        for kv in stage["kv_dtypes"]:
            if kv not in VALID_KV_DTYPES:
                raise ValueError(f"stage '{name}': unknown kv dtype '{kv}'")
        for spec in stage["speculation"]:
            if spec not in cfg.speculation:
                raise ValueError(
                    f"stage '{name}': speculation '{spec}' not defined in 'speculation:'"
                )
        wl = stage.get("workload", "synthetic")
        if wl not in VALID_WORKLOADS:
            raise ValueError(f"stage '{name}': unknown workload '{wl}'")
        if wl == "corpus" and "corpus_path" not in cfg.data:
            raise ValueError(f"stage '{name}' uses 'corpus' but data.corpus_path is unset")
        if wl == "sharegpt" and "sharegpt_path" not in cfg.data:
            raise ValueError(f"stage '{name}' uses 'sharegpt' but data.sharegpt_path is unset")


def expand_runs(
    cfg: ExperimentConfig, stage_names: list[str] | None = None
) -> list[ServerRun]:
    """Expand stages into the concrete list of server launches."""
    names = stage_names or list(cfg.stages)
    unknown = [n for n in names if n not in cfg.stages]
    if unknown:
        raise ValueError(f"unknown stage(s): {unknown}; available: {list(cfg.stages)}")

    runs: list[ServerRun] = []
    for name in names:
        stage = cfg.stages[name]
        ctxs = tuple(int(c) for c in stage["context_lengths"])
        concs = tuple(int(c) for c in stage["concurrency"])
        for kv in stage["kv_dtypes"]:
            for spec in stage["speculation"]:
                runs.append(
                    ServerRun(
                        stage=name,
                        kv_dtype=kv,
                        spec_name=spec,
                        spec_config=cfg.speculation[spec],
                        prefix_caching=bool(stage.get("prefix_caching", False)),
                        workload=stage.get("workload", "synthetic"),
                        workload_args=dict(stage.get("workload_args") or {}),
                        context_lengths=ctxs,
                        concurrency=concs,
                        # +16 leaves headroom for special tokens / template overhead
                        max_model_len=max(ctxs) + cfg.fixed.output_len + 16,
                        max_num_seqs=max(cfg.fixed.max_num_seqs, max(concs)),
                    )
                )
    return runs
