"""Quality evaluation of a running server (Topic 2: what does FP8 KV cost?).

Perplexity is computed THROUGH THE SERVER (vLLM's ``prompt_logprobs``), so it
exercises the real FP8 KV-cache kernels.  Computing it with HuggingFace
``transformers`` would not: HF does not use vLLM's FP8 cache path.

The same token windows (fixed by ``seed``) must be used for every KV dtype,
otherwise the FP16-vs-FP8 delta is dominated by which text was sampled.
Perplexity is computed per context length so we can see *where* any
degradation shows up.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import numpy as np


def build_windows(
    token_ids: list[int] | np.ndarray, context_len: int, n_windows: int, seed: int
) -> list[list[int]]:
    toks = np.asarray(token_ids, dtype=np.int64)
    if len(toks) <= context_len:
        raise ValueError(f"corpus ({len(toks)} tokens) shorter than context_len={context_len}")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(toks) - context_len, size=n_windows)
    return [toks[s : s + context_len].tolist() for s in starts]


def _window_nll(
    client: httpx.Client, model: str, ids: list[int]
) -> tuple[float, int] | None:
    payload = {
        "model": model,
        "prompt": ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": 0,  # vLLM extension: logprob of each actual prompt token
    }
    r = client.post("/v1/completions", json=payload)
    if r.status_code != 200:
        return None
    plp = r.json()["choices"][0].get("prompt_logprobs")
    if not plp:
        return None
    nll, n = 0.0, 0
    for tid, entry in zip(ids, plp):
        if entry is None:  # first token has no context to be predicted from
            continue
        lp = entry.get(str(tid))
        if lp is None:
            return None
        nll -= lp["logprob"]
        n += 1
    return nll, n


def perplexity_via_server(
    base_url: str,
    model: str,
    windows: list[list[int]],
    timeout_s: float = 600.0,
    workers: int = 4,
) -> dict[str, Any]:
    """Token-weighted perplexity over all windows."""
    with httpx.Client(
        base_url=base_url, timeout=timeout_s, trust_env=False
    ) as client, ThreadPoolExecutor(workers) as pool:
        outs = list(pool.map(lambda w: _window_nll(client, model, w), windows))
    good = [o for o in outs if o is not None]
    if not good:
        raise RuntimeError(
            "no usable prompt_logprobs came back; check the server supports "
            "'prompt_logprobs' on /v1/completions"
        )
    total_nll = sum(o[0] for o in good)
    total_n = sum(o[1] for o in good)
    return {
        "ppl": math.exp(total_nll / total_n),
        "mean_nll": total_nll / total_n,
        "n_tokens": total_n,
        "n_windows": len(good),
        "n_failed": len(outs) - len(good),
    }


def run_lm_eval(
    base_url: str,
    model: str,
    tasks: list[str],
    out_dir: str | Path,
    limit: int | None = None,
    num_concurrent: int = 16,
) -> dict[str, Any]:
    """Run MMLU / HumanEval etc. through lm-evaluation-harness against the server.

    Needs ``pip install lm-eval``.  HumanEval executes model-generated code, so
    it additionally needs ``--confirm_run_unsafe_code`` and HF_ALLOW_CODE_EVAL=1;
    we pass both when 'humaneval' is requested.  Run it in a sandbox you trust.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "lm_eval",
        "--model", "local-completions",
        "--model_args",
        f"model={model},base_url={base_url}/v1/completions,"
        f"num_concurrent={num_concurrent},tokenized_requests=False",
        "--tasks", ",".join(tasks),
        "--output_path", str(out_dir),
    ]
    env = dict(os.environ)
    if limit:
        cmd += ["--limit", str(limit)]
    if any("humaneval" in t for t in tasks):
        cmd += ["--confirm_run_unsafe_code"]
        env["HF_ALLOW_CODE_EVAL"] = "1"
    subprocess.run(cmd, check=True, env=env)

    result_files = sorted(out_dir.rglob("results*.json"), key=lambda p: p.stat().st_mtime)
    if not result_files:
        raise RuntimeError(f"lm_eval produced no results*.json under {out_dir}")
    return json.loads(result_files[-1].read_text(encoding="utf-8")).get("results", {})
