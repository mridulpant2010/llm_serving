"""Command-line interface.

    llm-serving list-runs  -c configs/experiments.yaml
    llm-serving run        -c configs/experiments.yaml --stages stage1_kv_capacity
    llm-serving quality    -c configs/experiments.yaml
    llm-serving analyze    --results results/results.jsonl --quality results/quality.jsonl

Use ``--backend mock`` to exercise the whole pipeline without a GPU.  Mock
output goes to ``results/mock_*.jsonl`` so it cannot be mistaken for real data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import expand_runs, load_config


def _default_out(backend: str, name: str) -> str:
    return f"results/{'mock_' if backend == 'mock' else ''}{name}.jsonl"


def _cmd_list_runs(args) -> None:
    cfg = load_config(args.config)
    runs = expand_runs(cfg, args.stages)
    total_levels = 0
    for r in runs:
        levels = len(r.context_lengths) * len(r.concurrency) * cfg.fixed.repetitions
        total_levels += levels
        print(f"{r.run_id:<62} workload={r.workload:<13} ctx={list(r.context_lengths)} "
              f"C={list(r.concurrency)}  ({levels} measurements)")
    print(f"\n{len(runs)} server launches, {total_levels} measurements in total")


def _cmd_run(args) -> None:
    from .orchestrator import run_experiments

    cfg = load_config(args.config)
    runs = expand_runs(cfg, args.stages)
    out = args.out or _default_out(args.backend, "results")
    print(f"{len(runs)} server launches -> {out}  (backend={args.backend})")
    run_experiments(cfg, runs, args.backend, out, resume=not args.no_resume,
                    log_dir=Path(out).parent / "server_logs")


def _cmd_quality(args) -> None:
    from .orchestrator import run_quality

    cfg = load_config(args.config)
    out = args.out or _default_out(args.backend, "quality")
    run_quality(cfg, args.backend, out, skip_lm_eval=args.skip_lm_eval,
                log_dir=Path(out).parent / "server_logs")


def _cmd_analyze(args) -> None:
    from .analysis import generate_report

    summary = generate_report(args.results, args.quality, args.out)
    print(f"Wrote report to {args.out}/summary.md")
    for r in summary["crossover"]:
        c = f"{r['c_star']:.1f}" if r["c_star"] else r["status"]
        print(f"  C*  kv={r['kv_dtype']:<5} spec={r['spec']:<12} ctx={r['context_len']:<5} -> {c}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="llm-serving", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("-c", "--config", default="configs/experiments.yaml")
        sp.add_argument("--backend", choices=["vllm", "mock"], default="vllm")

    sp = sub.add_parser("list-runs", help="show the server launches a config expands to")
    sp.add_argument("-c", "--config", default="configs/experiments.yaml")
    sp.add_argument("--stages", nargs="*")
    sp.set_defaults(fn=_cmd_list_runs)

    sp = sub.add_parser("run", help="run the throughput / latency sweeps")
    common(sp)
    sp.add_argument("--stages", nargs="*", help="default: all stages")
    sp.add_argument("--out")
    sp.add_argument("--no-resume", action="store_true", help="re-run rows already in --out")
    sp.set_defaults(fn=_cmd_run)

    sp = sub.add_parser("quality", help="perplexity (+ MMLU/HumanEval) per KV dtype")
    common(sp)
    sp.add_argument("--out")
    sp.add_argument("--skip-lm-eval", action="store_true")
    sp.set_defaults(fn=_cmd_quality)

    sp = sub.add_parser("analyze", help="tables, figures, C* from collected results")
    sp.add_argument("--results", default="results/results.jsonl")
    sp.add_argument("--quality", default=None)
    sp.add_argument("--out", default="results/report")
    sp.set_defaults(fn=_cmd_analyze)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
