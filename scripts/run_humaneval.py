#!/usr/bin/env python3
"""Run HumanEval evaluation for FP16 vs FP8 KV caches.

Usage:
    # 1. Direct vLLM engine mode (recommended, fastest, bypasses HTTP API limits):
    python scripts/run_humaneval.py --limit 20
    python scripts/run_humaneval.py                  # Full 164 problems

    # 2. Or against an already running vLLM server:
    python scripts/run_humaneval.py --mode server --base-url http://127.0.0.1:8000 --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def run_direct_vllm(
    model: str,
    kv_dtype: str,
    out_dir: Path,
    limit: int | None = None,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 4096,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_args = (
        f"pretrained={model},dtype=auto,kv_cache_dtype={kv_dtype},"
        f"gpu_memory_utilization={gpu_memory_utilization},max_model_len={max_model_len}"
    )
    cmd = [
        "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", "humaneval",
        "--confirm_run_unsafe_code",
        "--output_path", str(out_dir),
    ]
    if limit:
        cmd += ["--limit", str(limit)]

    env = dict(os.environ)
    env["HF_ALLOW_CODE_EVAL"] = "1"

    print(f"\n[HumanEval] Running direct vLLM for kv_dtype={kv_dtype} (limit={limit or 'full 164'})...")
    subprocess.run(cmd, check=True, env=env)

    result_files = sorted(out_dir.rglob("results*.json"), key=lambda p: p.stat().st_mtime)
    if not result_files:
        raise RuntimeError(f"No results*.json found in {out_dir}")
    return json.loads(result_files[-1].read_text(encoding="utf-8")).get("results", {})


def run_via_server(
    model: str,
    base_url: str,
    kv_dtype: str,
    out_dir: Path,
    limit: int | None = None,
    num_concurrent: int = 16,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Note: until is restricted to 4 stop sequences to adhere to OpenAI API max_length=4
    cmd = [
        "lm_eval",
        "--model", "local-completions",
        "--model_args",
        f"model={model},base_url={base_url}/v1/completions,"
        f"num_concurrent={num_concurrent},tokenized_requests=False",
        "--tasks", "humaneval",
        "--confirm_run_unsafe_code",
        "--gen_kwargs", "until=['\\nclass','\\ndef','\\n#','\\nif']",
        "--output_path", str(out_dir),
    ]
    if limit:
        cmd += ["--limit", str(limit)]

    env = dict(os.environ)
    env["HF_ALLOW_CODE_EVAL"] = "1"

    print(f"\n[HumanEval] Running via server {base_url} for kv_dtype={kv_dtype} (limit={limit or 'full 164'})...")
    subprocess.run(cmd, check=True, env=env)

    result_files = sorted(out_dir.rglob("results*.json"), key=lambda p: p.stat().st_mtime)
    if not result_files:
        raise RuntimeError(f"No results*.json found in {out_dir}")
    return json.loads(result_files[-1].read_text(encoding="utf-8")).get("results", {})


def extract_pass_at_1(results: dict) -> float | None:
    he = results.get("humaneval", {})
    for key in ("pass@1,create_test", "pass@1,none", "pass@1"):
        if key in he:
            return float(he[key])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HumanEval on Llama-3 with FP16 vs FP8 KV cache.")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model checkpoint")
    parser.add_argument("--mode", choices=["vllm", "server"], default="vllm", help="Execution mode")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="vLLM server base URL (for server mode)")
    parser.add_argument("--kv-dtypes", nargs="+", default=["auto", "fp8"], help="KV cache dtypes to test")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit (e.g. 20; omit for full 164)")
    parser.add_argument("--out-dir", default="results_budget/lm_eval", help="Directory to save eval outputs")
    parser.add_argument("--update-quality", default="results_budget/quality.jsonl", help="Path to quality.jsonl to append records")
    args = parser.parse_args()

    base_out = Path(args.out_dir)
    scores = {}

    for kv in args.kv_dtypes:
        kv_dir = base_out / kv
        if args.mode == "vllm":
            res = run_direct_vllm(args.model, kv, kv_dir, limit=args.limit)
        else:
            res = run_via_server(args.model, args.base_url, kv, kv_dir, limit=args.limit)

        score = extract_pass_at_1(res)
        scores[kv] = score
        print(f"--> KV {kv}: HumanEval pass@1 = {score:.4f}" if score is not None else f"--> KV {kv}: Evaluation finished")

        if args.update_quality:
            qpath = Path(args.update_quality)
            record = {
                "kind": "lm_eval",
                "kv_dtype": kv,
                "backend": args.mode,
                "model": args.model,
                "results": res,
                "timestamp": time.time(),
            }
            with qpath.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(f"    Appended record to {qpath}")

    # Final summary table
    print("\n" + "=" * 55)
    print("           HUMANEVAL EVALUATION SUMMARY")
    print("=" * 55)
    for kv, score in scores.items():
        s_str = f"{score * 100:.2f}%" if score is not None else "N/A"
        print(f"  KV Cache: {kv:<6} | pass@1: {s_str}")

    if "auto" in scores and "fp8" in scores and scores["auto"] is not None and scores["fp8"] is not None:
        p_auto = scores["auto"]
        p_fp8 = scores["fp8"]
        delta = p_fp8 - p_auto
        delta_pct = (delta / p_auto) * 100 if p_auto > 0 else 0
        sign = "+" if delta >= 0 else ""
        print("-" * 55)
        print(f"  Delta (FP8 vs FP16): {sign}{delta * 100:.2f}% ({sign}{delta_pct:.2f}% relative)")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
