"""Run controller: executes the experiment matrix and writes results as JSONL.

Each line of the results file is one (server run, context length, concurrency,
repetition) measurement.  The file is append-only and flushed after every row,
so a crash (or a preempted cluster job) loses at most one measurement, and
``--resume`` skips rows that already exist.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

from . import quality_eval
from .config import ExperimentConfig, ServerRun
from .load_client import run_closed_loop, summarize
from .mock_server import MockServer
from .server import VllmServer
from .telemetry import GpuSampler, MetricsScraper, get_cache_info
from .workload import TokenizerFn, build_workload

Log = Callable[[str], None]


@contextmanager
def make_server(backend: str, cfg: ExperimentConfig, run: ServerRun, log_dir: str | Path):
    if backend == "vllm":
        with VllmServer(cfg, run, log_dir=log_dir) as srv:
            yield srv
    elif backend == "mock":
        with MockServer(run) as srv:
            yield srv
    else:
        raise ValueError(f"unknown backend '{backend}'")


def _key(row: dict) -> tuple:
    return (
        row.get("stage"), row.get("kv_dtype"), row.get("spec"), row.get("prefix_caching"),
        row.get("context_len"), row.get("concurrency"), row.get("rep"),
    )


def _load_done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("status") == "ok":
                done.add(_key(row))
    return done


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def _tokenizer_for(cfg: ExperimentConfig, runs: list[ServerRun]):
    """Only load the (heavy) tokenizer if some stage actually needs it."""
    if not any(r.workload in ("corpus", "sharegpt") for r in runs):
        return None
    from .workload import hf_tokenizer_fn

    return hf_tokenizer_fn(cfg.tokenizer_name)


def run_experiments(
    cfg: ExperimentConfig,
    runs: list[ServerRun],
    backend: str,
    out_path: str | Path,
    resume: bool = True,
    tokenizer_fn: TokenizerFn | None = None,
    log: Log = print,
    log_dir: str | Path = "results/server_logs",
) -> None:
    out_path = Path(out_path)
    done = _load_done(out_path) if resume else set()
    if tokenizer_fn is None:
        tokenizer_fn = _tokenizer_for(cfg, runs)

    for i, run in enumerate(runs, 1):
        log(f"\n=== [{i}/{len(runs)}] {run.run_id} ===")
        workload = build_workload(run.workload, cfg.data, run.workload_args, tokenizer_fn)
        try:
            with make_server(backend, cfg, run, log_dir) as server:
                cache = get_cache_info(server.base_url)
                log(f"KV pool: {cache}")
                _sweep(cfg, run, backend, server.base_url, workload, cache, GpuSampler(),
                       out_path, done, log)
        except Exception as e:  # keep going: one bad config shouldn't lose the rest
            log(f"!! run failed: {type(e).__name__}: {e}")
            _append(out_path, {
                "status": "server_failed", "stage": run.stage, "kv_dtype": run.kv_dtype,
                "spec": run.spec_name, "prefix_caching": run.prefix_caching,
                "backend": backend, "error": f"{type(e).__name__}: {e}"[:500],
                "timestamp": time.time(),
            })


WARMUP_SEED = 999_999  # fixed, and distinct from every measurement seed


def _sweep(cfg, run, backend, base_url, workload, cache, gpu, out_path, done, log) -> None:
    fx = cfg.fixed

    # Warm-up: CUDA graphs, kernel autotuning, allocator growth. Discarded.
    warm = workload.make_requests(
        fx.warmup_requests, min(run.context_lengths), fx.output_len, seed=WARMUP_SEED
    )
    asyncio.run(run_closed_loop(base_url, cfg.model, warm, min(8, len(warm)), fx.request_timeout_s))

    for ctx in run.context_lengths:
        # KV-pool-limited concurrency from first principles:
        #   (blocks * tokens_per_block) / tokens_per_request
        analytic = None
        if cache.get("num_gpu_blocks") and cache.get("block_size"):
            analytic = int(cache["num_gpu_blocks"] * cache["block_size"] // (ctx + fx.output_len))

        for conc in run.concurrency:
            n_req = max(fx.min_requests_per_level, conc * fx.rounds_per_level)
            saturated = False
            for rep in range(fx.repetitions):
                base = {
                    "stage": run.stage, "kv_dtype": run.kv_dtype, "spec": run.spec_name,
                    "prefix_caching": run.prefix_caching, "context_len": ctx,
                    "concurrency": conc, "rep": rep,
                }
                if _key(base) in done:
                    continue
                requests = workload.make_requests(
                    n_req, ctx, fx.output_len, seed=fx.seed + 1000 * rep + conc
                )
                scraper = MetricsScraper(base_url)
                scraper.start()
                gpu.start()
                load = asyncio.run(
                    run_closed_loop(base_url, cfg.model, requests, conc, fx.request_timeout_s)
                )
                gpu_stats = gpu.stop()
                srv_stats = scraper.stop()

                row = {
                    **base, "status": "ok", "backend": backend, "model": cfg.model,
                    "workload": run.workload, "output_len": fx.output_len,
                    "kv_blocks": cache.get("num_gpu_blocks"),
                    "block_size": cache.get("block_size"),
                    "max_concurrency_analytic": analytic,
                    **summarize(load), **srv_stats, **gpu_stats,
                    "timestamp": time.time(),
                }
                _append(out_path, row)
                log(
                    f"ctx={ctx:>5} C={conc:>4} rep={rep} "
                    f"tput={row['throughput_tps'] or 0:8.1f} tok/s  "
                    f"TPOT p50={row['tpot_ms_p50'] or float('nan'):7.1f} ms  "
                    f"fail={row['num_failed']}/{row['num_requests']}"
                )
                if row["num_failed"] / max(1, row["num_requests"]) >= fx.stop_on_failure_fraction:
                    saturated = True
                    break
            if saturated:
                log(f"   >={fx.stop_on_failure_fraction:.0%} failures at C={conc}; "
                    f"skipping higher C for ctx={ctx}")
                break


def run_quality(
    cfg: ExperimentConfig,
    backend: str,
    out_path: str | Path,
    tokenizer_fn: TokenizerFn | None = None,
    skip_lm_eval: bool = False,
    log: Log = print,
    log_dir: str | Path = "results/server_logs",
) -> None:
    """Stage 2: perplexity (per context length) and task accuracy per KV dtype."""
    q = cfg.quality
    if not q:
        raise ValueError("config has no 'quality:' section")
    out_path = Path(out_path)
    ctxs = [int(c) for c in q["context_lengths"]]

    if tokenizer_fn is None:
        from .workload import hf_tokenizer_fn

        tokenizer_fn = hf_tokenizer_fn(cfg.tokenizer_name)
    corpus = tokenizer_fn(Path(q["corpus_path"]).read_text(encoding="utf-8"))
    windows = {
        c: quality_eval.build_windows(corpus, c, int(q.get("num_windows", 64)), cfg.fixed.seed)
        for c in ctxs
    }

    for kv in q["kv_dtypes"]:
        run = ServerRun(
            stage="quality", kv_dtype=kv, spec_name="none", spec_config=None,
            prefix_caching=False, workload="corpus", workload_args={},
            context_lengths=tuple(ctxs), concurrency=(1,),
            max_model_len=max(ctxs) + 16, max_num_seqs=cfg.fixed.max_num_seqs,
        )
        log(f"\n=== quality: kv_dtype={kv} ===")
        with make_server(backend, cfg, run, log_dir) as server:
            for c in ctxs:
                res = quality_eval.perplexity_via_server(server.base_url, cfg.model, windows[c])
                log(f"ctx={c:>5}  PPL={res['ppl']:.4f}  ({res['n_tokens']} tokens)")
                _append(out_path, {
                    "kind": "perplexity", "kv_dtype": kv, "context_len": c,
                    "backend": backend, "model": cfg.model, "timestamp": time.time(), **res,
                })
            tasks = q.get("lm_eval_tasks") or []
            if tasks and not skip_lm_eval and backend == "vllm":
                results = quality_eval.run_lm_eval(
                    server.base_url, cfg.model, tasks,
                    out_dir=Path(out_path).parent / "lm_eval" / kv,
                    limit=q.get("lm_eval_limit"),
                )
                _append(out_path, {
                    "kind": "lm_eval", "kv_dtype": kv, "backend": backend,
                    "model": cfg.model, "results": results, "timestamp": time.time(),
                })
