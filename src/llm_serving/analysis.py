"""Analysis: turn raw results into capacity tables, speedup curves, C*, regime maps.

Pure standard library + numpy for the maths (so it is easy to test); matplotlib
is imported lazily and only for the figure functions.

Definitions
-----------
Speedup ratio   S(C) = TPOT_p50(no speculation, C) / TPOT_p50(speculation, C)
Crossover C*    the concurrency where S falls through 1.0 (linear interpolation
                in log2(C) between the two bracketing sweep points).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

KEY_FIELDS = ("kv_dtype", "spec", "prefix_caching", "workload", "context_len", "concurrency")


# ---------------------------------------------------------------------------
# Loading and aggregation
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_results(path: str | Path) -> list[dict]:
    return [r for r in load_jsonl(path) if r.get("status") == "ok"]


def aggregate(rows: list[dict]) -> list[dict]:
    """Median over repetitions for every numeric field, per configuration."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in KEY_FIELDS)].append(r)
    out = []
    for key, grp in groups.items():
        agg = dict(zip(KEY_FIELDS, key))
        agg["n_reps"] = len(grp)
        for field in grp[0]:
            if field in KEY_FIELDS or field in ("rep", "timestamp"):
                continue
            vals = [g[field] for g in grp if isinstance(g.get(field), (int, float))
                    and not isinstance(g.get(field), bool)]
            if vals:
                agg[field] = float(median(vals))
            elif field not in agg:
                agg[field] = grp[0][field]
        out.append(agg)
    return sorted(out, key=lambda a: tuple(str(a[k]) for k in KEY_FIELDS[:-2]) + (a["context_len"], a["concurrency"]))


def _select(agg: list[dict], **conds) -> list[dict]:
    return sorted(
        (a for a in agg if all(a.get(k) == v for k, v in conds.items())),
        key=lambda a: a["concurrency"],
    )


# ---------------------------------------------------------------------------
# Topic 2: capacity and throughput
# ---------------------------------------------------------------------------

def capacity_table(agg: list[dict]) -> list[dict]:
    """Per (kv_dtype, context): KV blocks, analytic ceiling, observed onset of pressure."""
    seen: dict[tuple, dict] = {}
    for a in agg:
        if a["spec"] != "none":
            continue
        key = (a["kv_dtype"], a["workload"], a["context_len"])
        row = seen.setdefault(key, {
            "kv_dtype": a["kv_dtype"], "workload": a["workload"], "context_len": a["context_len"],
            "kv_blocks": a.get("kv_blocks"), "block_size": a.get("block_size"),
            "max_concurrency_analytic": a.get("max_concurrency_analytic"),
            "first_pressure_C": None,
        })
        pressured = (a.get("srv_waiting_max") or 0) > 0 or (a.get("srv_preemptions_delta") or 0) > 0
        if pressured and (row["first_pressure_C"] is None or a["concurrency"] < row["first_pressure_C"]):
            row["first_pressure_C"] = a["concurrency"]
    return sorted(seen.values(), key=lambda r: (r["workload"], r["context_len"], r["kv_dtype"]))


def capacity_gain(table: list[dict], base: str = "auto", other: str = "fp8") -> list[dict]:
    """How much extra concurrency ``other`` buys over ``base`` at each context length."""
    by = {(r["workload"], r["context_len"], r["kv_dtype"]): r for r in table}
    out = []
    for (wl, ctx, kv), r in by.items():
        if kv != base or (wl, ctx, other) not in by:
            continue
        o = by[(wl, ctx, other)]
        a0, a1 = r["max_concurrency_analytic"], o["max_concurrency_analytic"]
        out.append({
            "workload": wl, "context_len": ctx,
            "analytic_base": a0, "analytic_other": a1,
            "analytic_gain_x": (a1 / a0) if a0 and a1 else None,
            "pressure_C_base": r["first_pressure_C"], "pressure_C_other": o["first_pressure_C"],
        })
    return sorted(out, key=lambda r: (r["workload"], r["context_len"]))


def throughput_knee(series: list[dict], tol: float = 0.05) -> dict[str, Any]:
    """Peak throughput and the smallest C that reaches within ``tol`` of it.

    A knee well below the memory ceiling means the extra concurrency is lost to
    a compute bottleneck rather than turned into throughput.
    """
    pts = [(a["concurrency"], a["throughput_tps"]) for a in series if a.get("throughput_tps")]
    if not pts:
        return {"peak_tps": None, "knee_C": None}
    peak = max(t for _, t in pts)
    knee = min(c for c, t in pts if t >= (1 - tol) * peak)
    return {"peak_tps": peak, "knee_C": knee}


# ---------------------------------------------------------------------------
# Topic 3: speedup and crossover
# ---------------------------------------------------------------------------

@dataclass
class Crossover:
    c_star: float | None
    status: str  # crosses | no_crossover_in_range | never_helps | insufficient


def speedup_series(
    agg: list[dict], kv_dtype: str, spec: str, ctx: int, workload: str,
    prefix_caching: bool = False, metric: str = "tpot_ms_p50",
) -> tuple[list[int], list[float]]:
    common = dict(kv_dtype=kv_dtype, context_len=ctx, workload=workload,
                  prefix_caching=prefix_caching)
    base = {a["concurrency"]: a.get(metric) for a in _select(agg, spec="none", **common)}
    sp = {a["concurrency"]: a.get(metric) for a in _select(agg, spec=spec, **common)}
    cs = sorted(c for c in base if c in sp and base[c] and sp[c])
    return cs, [base[c] / sp[c] for c in cs]


def find_crossover(concurrency: list[int], speedup: list[float]) -> Crossover:
    if len(concurrency) < 2:
        return Crossover(None, "insufficient")
    if all(s < 1.0 for s in speedup):
        return Crossover(None, "never_helps")
    for i in range(len(speedup) - 1):
        s0, s1 = speedup[i], speedup[i + 1]
        if s0 >= 1.0 > s1:
            x0, x1 = math.log2(concurrency[i]), math.log2(concurrency[i + 1])
            x = x0 + (s0 - 1.0) / (s0 - s1) * (x1 - x0)
            return Crossover(2 ** x, "crosses")
    return Crossover(None, "no_crossover_in_range")  # C* is beyond the largest C tested


def crossover_table(agg: list[dict]) -> list[dict]:
    rows = []
    combos = {(a["kv_dtype"], a["spec"], a["context_len"], a["workload"], a["prefix_caching"])
              for a in agg if a["spec"] != "none"}
    for kv, spec, ctx, wl, pc in sorted(combos, key=str):
        cs, ss = speedup_series(agg, kv, spec, ctx, wl, pc)
        if not cs:
            continue
        co = find_crossover(cs, ss)
        acc = [a.get("srv_acceptance_rate") for a in _select(
            agg, kv_dtype=kv, spec=spec, context_len=ctx, workload=wl, prefix_caching=pc)
            if a.get("srv_acceptance_rate") is not None]
        rows.append({
            "kv_dtype": kv, "spec": spec, "context_len": ctx, "workload": wl,
            "speedup_at_min_C": ss[0], "min_C": cs[0],
            "speedup_at_max_C": ss[-1], "max_C": cs[-1],
            "c_star": co.c_star, "status": co.status,
            "mean_acceptance_rate": float(np.mean(acc)) if acc else None,
        })
    return rows


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

def quality_deltas(quality_rows: list[dict], base: str = "auto", other: str = "fp8") -> list[dict]:
    ppl = {(r["kv_dtype"], r["context_len"]): r["ppl"]
           for r in quality_rows if r.get("kind") == "perplexity"}
    out = []
    for (kv, ctx), p0 in sorted(ppl.items()):
        if kv != base or (other, ctx) not in ppl:
            continue
        p1 = ppl[(other, ctx)]
        out.append({"context_len": ctx, f"ppl_{base}": p0, f"ppl_{other}": p1,
                    "delta": p1 - p0, "delta_pct": 100.0 * (p1 - p0) / p0})
    return out


# ---------------------------------------------------------------------------
# Figures (matplotlib imported lazily)
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _label(kv: str) -> str:
    return "FP8 KV" if kv.startswith("fp8") else "FP16/BF16 KV"


def fig_throughput(agg: list[dict], ctx: int, workload: str, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    for kv in sorted({a["kv_dtype"] for a in agg}):
        s = _select(agg, kv_dtype=kv, spec="none", context_len=ctx, workload=workload,
                    prefix_caching=False)
        if s:
            ax.plot([a["concurrency"] for a in s], [a["throughput_tps"] for a in s],
                    "o-", label=_label(kv))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Concurrency C"); ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title(f"Throughput vs concurrency (context={ctx})")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_speedup(agg: list[dict], kv: str, ctx: int, workload: str, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    for spec in sorted({a["spec"] for a in agg} - {"none"}):
        cs, ss = speedup_series(agg, kv, spec, ctx, workload)
        if not cs:
            continue
        line, = ax.plot(cs, ss, "o-", label=spec)
        co = find_crossover(cs, ss)
        if co.c_star:
            ax.axvline(co.c_star, color=line.get_color(), ls=":", alpha=0.6)
            ax.annotate(f"C*≈{co.c_star:.0f}", (co.c_star, 1.0), textcoords="offset points",
                        xytext=(4, 6), color=line.get_color(), fontsize=9)
    ax.axhline(1.0, color="k", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Concurrency C"); ax.set_ylabel("Speedup S(C) = TPOT_base / TPOT_spec")
    ax.set_title(f"Speculative decoding speedup ({_label(kv)}, context={ctx})")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_cstar_shift(table: list[dict], path: Path) -> None:
    """Bar chart: C* per speculation method, FP16 vs FP8 (the regime-map headline)."""
    plt = _plt()
    specs = sorted({r["spec"] for r in table})
    kvs = sorted({r["kv_dtype"] for r in table})
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / max(1, len(kvs))
    for j, kv in enumerate(kvs):
        vals = []
        for spec in specs:
            hit = [r for r in table if r["spec"] == spec and r["kv_dtype"] == kv]
            r = hit[0] if hit else None
            vals.append(r["c_star"] if r and r["c_star"] else 0)
        ax.bar(np.arange(len(specs)) + j * width, vals, width, label=_label(kv))
    ax.set_xticks(np.arange(len(specs)) + width * (len(kvs) - 1) / 2, specs)
    ax.set_ylabel("Crossover C*  (0 = no crossover found)")
    ax.set_title("Where speculation stops helping, by KV precision")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_regime_map(agg: list[dict], spec: str, workload: str, path: Path) -> None:
    """Heatmap of S over (context, C), one panel per KV dtype; contour at S=1."""
    plt = _plt()
    kvs = sorted({a["kv_dtype"] for a in agg})
    ctxs = sorted({a["context_len"] for a in agg if a["workload"] == workload})
    concs = sorted({a["concurrency"] for a in agg if a["workload"] == workload})
    fig, axes = plt.subplots(1, len(kvs), figsize=(6 * len(kvs), 4.5), squeeze=False)
    for ax, kv in zip(axes[0], kvs):
        grid = np.full((len(ctxs), len(concs)), np.nan)
        for i, ctx in enumerate(ctxs):
            cs, ss = speedup_series(agg, kv, spec, ctx, workload)
            for c, s in zip(cs, ss):
                grid[i, concs.index(c)] = s
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0.5, vmax=2.0)
        if np.isfinite(grid).sum() > 3 and grid.shape[0] > 1 and grid.shape[1] > 1:
            ax.contour(np.nan_to_num(grid, nan=1.0), levels=[1.0], colors="k")
        ax.set_xticks(range(len(concs)), concs); ax.set_yticks(range(len(ctxs)), ctxs)
        ax.set_xlabel("Concurrency C"); ax.set_ylabel("Context length")
        ax.set_title(f"{spec}: {_label(kv)}")
        fig.colorbar(im, ax=ax, label="speedup S")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _md_table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_no data_\n"
    def fmt(v):
        if v is None: return "-"
        if isinstance(v, float): return f"{v:.3g}"
        return str(v)
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    return head + "".join("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |\n" for r in rows)


def generate_report(
    results_path: str | Path, quality_path: str | Path | None, out_dir: str | Path
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = load_results(results_path)
    if not raw:
        raise ValueError(f"no successful rows in {results_path}")
    backends = sorted({r.get("backend") for r in raw})
    agg = aggregate(raw)

    cap = capacity_table(agg)
    gain = capacity_gain(cap)
    cross = crossover_table(agg)
    qdelta = quality_deltas(load_jsonl(quality_path)) if quality_path else []

    knees = []
    for kv, wl, ctx in sorted({(a["kv_dtype"], a["workload"], a["context_len"]) for a in agg
                               if a["spec"] == "none"}, key=str):
        k = throughput_knee(_select(agg, kv_dtype=kv, spec="none", workload=wl,
                                    context_len=ctx, prefix_caching=False))
        knees.append({"kv_dtype": kv, "workload": wl, "context_len": ctx, **k})

    # figures
    for wl, ctx in sorted({(a["workload"], a["context_len"]) for a in agg}):
        if any(a["spec"] == "none" and a["workload"] == wl and a["context_len"] == ctx for a in agg):
            fig_throughput(agg, ctx, wl, out / f"throughput_{wl}_ctx{ctx}.png")
    for kv, ctx, wl in sorted({(r["kv_dtype"], r["context_len"], r["workload"]) for r in cross}, key=str):
        fig_speedup(agg, kv, ctx, wl, out / f"speedup_{wl}_{kv}_ctx{ctx}.png")
    if cross:
        fig_cstar_shift(cross, out / "cstar_shift.png")
        for spec in sorted({r["spec"] for r in cross}):
            wl = next(r["workload"] for r in cross if r["spec"] == spec)
            fig_regime_map(agg, spec, wl, out / f"regime_map_{spec}.png")

    summary = {"backends": backends, "capacity": cap, "capacity_gain": gain,
               "throughput_knee": knees, "crossover": cross, "quality_delta": qdelta}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    banner = ""
    if "mock" in backends:
        banner = ("> **WARNING: these results come from the MOCK server (simulated). "
                  "Do not report them as measurements.**\n\n")
    md = (
        f"# Experiment summary\n\n{banner}"
        "## KV capacity (FP16 vs FP8)\n" + _md_table(cap, ["workload", "context_len", "kv_dtype",
            "kv_blocks", "max_concurrency_analytic", "first_pressure_C"]) +
        "\n### Capacity gain\n" + _md_table(gain, ["workload", "context_len", "analytic_base",
            "analytic_other", "analytic_gain_x", "pressure_C_base", "pressure_C_other"]) +
        "\n## Throughput knee (turns concurrency into throughput, or hits a compute wall?)\n" +
        _md_table(knees, ["workload", "context_len", "kv_dtype", "peak_tps", "knee_C"]) +
        "\n## Speculative decoding crossover C*\n" + _md_table(cross, ["workload", "kv_dtype",
            "spec", "context_len", "speedup_at_min_C", "speedup_at_max_C", "c_star", "status",
            "mean_acceptance_rate"]) +
        "\n## Quality: FP8 vs FP16 KV cache\n" + _md_table(qdelta, list(qdelta[0]) if qdelta else [])
    )
    (out / "summary.md").write_text(md, encoding="utf-8")
    return summary
