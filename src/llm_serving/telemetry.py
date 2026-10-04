"""Telemetry: server-side Prometheus metrics and GPU sampling.

vLLM metric names have changed between releases (e.g. ``gpu_cache_usage_perc``
became ``kv_cache_usage_perc``).  Each quantity below therefore has a list of
candidate names; the first one present wins.  If something comes back ``None``
on your server, run ``curl localhost:8000/metrics`` and add the real name to
the candidate list.

GPU note: NVML's ``utilization.gpu`` is the fraction of time *any* kernel was
running, not how many SMs were busy.  It is a coarse proxy for "compute is
saturated"; use Nsight/DCGM if you need true SM occupancy.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any

import httpx

_LINE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+]?(?:[0-9.]+(?:[eE][-+]?[0-9]+)?|nan|inf))\s*$",
    re.IGNORECASE,
)
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

Sample = tuple[str, dict[str, str], float]


def parse_prometheus(text: str) -> list[Sample]:
    """Parse Prometheus text exposition format into (name, labels, value)."""
    samples: list[Sample] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, label_blob, value = m.groups()
        labels = dict(_LABEL.findall(label_blob)) if label_blob else {}
        try:
            samples.append((name, labels, float(value)))
        except ValueError:
            continue
    return samples


def _norm(name: str) -> str:
    for suffix in ("_total", "_created"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _first(samples: list[Sample], candidates: list[str]) -> float | None:
    wanted = [_norm(c) for c in candidates]
    for cand in wanted:  # candidate order = priority
        total, found = 0.0, False
        for name, _labels, value in samples:
            if _norm(name) == cand:
                total += value
                found = True
        if found:
            return total
    return None


GAUGES = {
    "kv_usage": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
    "running": ["vllm:num_requests_running"],
    "waiting": ["vllm:num_requests_waiting"],
}
COUNTERS = {
    "preemptions": ["vllm:num_preemptions"],
    "draft_tokens": ["vllm:spec_decode_num_draft_tokens"],
    "accepted_tokens": ["vllm:spec_decode_num_accepted_tokens"],
    "num_drafts": ["vllm:spec_decode_num_drafts"],
}


def _fetch(client: httpx.Client, base_url: str) -> list[Sample] | None:
    try:
        r = client.get(f"{base_url}/metrics")
        r.raise_for_status()
        return parse_prometheus(r.text)
    except httpx.HTTPError:
        return None


def get_cache_info(base_url: str) -> dict[str, Any]:
    """KV pool size from vLLM's ``cache_config_info`` metric."""
    info: dict[str, Any] = {"num_gpu_blocks": None, "block_size": None, "cache_dtype": None}
    with httpx.Client(timeout=10.0, trust_env=False) as client:
        samples = _fetch(client, base_url)
    for name, labels, _ in samples or []:
        if _norm(name).endswith("cache_config_info"):
            if "num_gpu_blocks" in labels:
                info["num_gpu_blocks"] = int(labels["num_gpu_blocks"])
            if "block_size" in labels:
                info["block_size"] = int(labels["block_size"])
            info["cache_dtype"] = labels.get("cache_dtype")
            break
    return info


class MetricsScraper:
    """Polls ``/metrics`` in a background thread while a load level runs."""

    def __init__(self, base_url: str, interval_s: float = 0.5):
        self.base_url, self.interval_s = base_url, interval_s
        self._series: list[list[Sample]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._client = httpx.Client(timeout=5.0, trust_env=False)

    def _grab(self) -> None:
        s = _fetch(self._client, self.base_url)
        if s is not None:
            self._series.append(s)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._grab()

    def start(self) -> None:
        self._series, self._stop = [], threading.Event()
        self._grab()  # baseline for counter deltas
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._grab()  # final reading for counter deltas
        self._client.close()
        return self.summary()

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, cands in GAUGES.items():
            vals = [v for s in self._series if (v := _first(s, cands)) is not None]
            out[f"srv_{key}_max"] = max(vals) if vals else None
            out[f"srv_{key}_mean"] = sum(vals) / len(vals) if vals else None
        for key, cands in COUNTERS.items():
            first = _first(self._series[0], cands) if self._series else None
            last = _first(self._series[-1], cands) if self._series else None
            out[f"srv_{key}_delta"] = (
                None if first is None or last is None else last - first
            )
        drafts, acc = out["srv_draft_tokens_delta"], out["srv_accepted_tokens_delta"]
        out["srv_acceptance_rate"] = acc / drafts if drafts and acc is not None else None
        nd = out["srv_num_drafts_delta"]
        # tokens emitted per verify step = 1 bonus/corrected token + accepted drafts
        out["srv_mean_accept_len"] = 1 + acc / nd if nd and acc is not None else None
        return out


class GpuSampler:
    """Samples GPU utilization / memory / power via NVML (optional)."""

    def __init__(self, device_index: int = 0, interval_s: float = 0.1):
        self.interval_s = interval_s
        self._samples: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._nvml = pynvml
        except Exception:  # no driver, no pynvml, no GPU: telemetry is optional
            self._nvml = None

    @property
    def available(self) -> bool:
        return self._nvml is not None

    def _loop(self) -> None:
        nv, h = self._nvml, self._handle
        while not self._stop.wait(self.interval_s):
            try:
                util = nv.nvmlDeviceGetUtilizationRates(h).gpu
                mem = nv.nvmlDeviceGetMemoryInfo(h).used / (1024 * 1024)
                power = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
                self._samples.append((float(util), mem, power))
            except Exception:
                continue

    def start(self) -> None:
        self._samples, self._stop = [], threading.Event()
        if self.available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        if not self._samples:
            return {"gpu_util_pct_mean": None, "gpu_mem_used_mb_max": None, "gpu_power_w_mean": None}
        utils, mems, powers = zip(*self._samples)
        return {
            "gpu_util_pct_mean": sum(utils) / len(utils),
            "gpu_mem_used_mb_max": max(mems),
            "gpu_power_w_mean": sum(powers) / len(powers),
        }
