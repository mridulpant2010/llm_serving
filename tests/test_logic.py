"""Unit tests for the pure-logic parts: config, workloads, telemetry parsing, analysis maths."""

import math

import numpy as np
import pytest

from llm_serving import analysis, config, telemetry, workload


# ---------------------------------------------------------------- config ----

def _cfg(tmp_path, stages=None, extra=""):
    y = f"""
model: m
speculation:
  none: null
  ngram: {{method: ngram, num_speculative_tokens: 4}}
fixed: {{output_len: 32, max_num_seqs: 64}}
{extra}
stages:
  s1:
    kv_dtypes: [auto, fp8]
    speculation: [none, ngram]
    context_lengths: [128, 512]
    concurrency: [1, 128]
"""
    p = tmp_path / "c.yaml"
    p.write_text(y)
    return config.load_config(p)


def test_expand_runs_is_kv_times_spec(tmp_path):
    runs = config.expand_runs(_cfg(tmp_path))
    assert len(runs) == 4
    assert {(r.kv_dtype, r.spec_name) for r in runs} == {
        ("auto", "none"), ("auto", "ngram"), ("fp8", "none"), ("fp8", "ngram")}


def test_max_model_len_and_max_num_seqs(tmp_path):
    r = config.expand_runs(_cfg(tmp_path))[0]
    assert r.max_model_len == 512 + 32 + 16          # largest ctx + output + headroom
    assert r.max_num_seqs == 128                     # raised: stage sweeps C=128 > configured 64


def test_unknown_spec_in_stage_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("model: m\nstages:\n  s:\n    kv_dtypes: [auto]\n    speculation: [eagle]\n"
                 "    context_lengths: [1]\n    concurrency: [1]\n")
    with pytest.raises(ValueError, match="eagle"):
        config.load_config(p)


def test_typo_in_fixed_section_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("model: m\nfixed: {outptu_len: 5}\n")
    with pytest.raises(ValueError, match="fixed"):
        config.load_config(p)


def test_repo_experiment_yaml_is_valid():
    cfg = config.load_config("configs/experiments.yaml")
    assert len(config.expand_runs(cfg)) == 2 + 4 + 4 + 2


# -------------------------------------------------------------- workload ----

def test_synthetic_exact_length_and_no_shared_prefix():
    reqs = workload.SyntheticWorkload().make_requests(8, 64, 16, seed=1)
    assert all(len(r.prompt_token_ids) == 64 and r.max_tokens == 16 for r in reqs)
    assert len({tuple(r.prompt_token_ids[:8]) for r in reqs}) == 8


def test_shared_prefix_fraction():
    reqs = workload.SharedPrefixWorkload(0.5).make_requests(6, 100, 8, seed=2)
    assert len({tuple(r.prompt_token_ids[:50]) for r in reqs}) == 1
    assert len({tuple(r.prompt_token_ids[50:]) for r in reqs}) == 6


def test_corpus_windows_are_slices_of_corpus():
    corpus = list(range(1000))
    reqs = workload.CorpusWorkload(corpus).make_requests(5, 50, 8, seed=3)
    for r in reqs:
        s = r.prompt_token_ids[0]
        assert r.prompt_token_ids == corpus[s : s + 50]


def test_corpus_too_short_raises():
    with pytest.raises(ValueError):
        workload.CorpusWorkload([1, 2, 3]).make_requests(1, 10, 1, 0)


def test_workload_is_deterministic_per_seed():
    a = workload.SyntheticWorkload().make_requests(3, 20, 4, seed=7)
    b = workload.SyntheticWorkload().make_requests(3, 20, 4, seed=7)
    assert [r.prompt_token_ids for r in a] == [r.prompt_token_ids for r in b]


# ------------------------------------------------------------- telemetry ----

PROM = """# HELP vllm:num_requests_running running
vllm:num_requests_running{model_name="m"} 12.0
vllm:kv_cache_usage_perc{model_name="m"} 0.75
vllm:num_preemptions_total{model_name="m"} 3.0
vllm:cache_config_info{block_size="16",cache_dtype="fp8",num_gpu_blocks="9000"} 1.0
"""


def test_parse_prometheus_labels_and_values():
    s = telemetry.parse_prometheus(PROM)
    names = {n for n, _, _ in s}
    assert "vllm:kv_cache_usage_perc" in names
    info = next(l for n, l, _ in s if n == "vllm:cache_config_info")
    assert info["num_gpu_blocks"] == "9000" and info["cache_dtype"] == "fp8"


def test_first_prefers_listed_candidate_and_ignores_total_suffix():
    s = telemetry.parse_prometheus(PROM)
    assert telemetry._first(s, telemetry.GAUGES["kv_usage"]) == 0.75   # new-style name
    assert telemetry._first(s, telemetry.COUNTERS["preemptions"]) == 3.0  # *_total matched
    assert telemetry._first(s, telemetry.COUNTERS["draft_tokens"]) is None


# -------------------------------------------------------------- analysis ----

def test_find_crossover_interpolates_in_log2():
    # S goes 2.0 at C=8 -> 0.5 at C=32. S=1 at 1/3 of the way in S (linear), so
    # log2 position = 3 + (2-1)/(2-0.5) * 2 = 4.333
    co = analysis.find_crossover([8, 32], [2.0, 0.5])
    assert co.status == "crosses"
    assert co.c_star == pytest.approx(2 ** (3 + (1.0 / 1.5) * 2))


def test_find_crossover_statuses():
    assert analysis.find_crossover([1, 2, 4], [0.9, 0.8, 0.7]).status == "never_helps"
    assert analysis.find_crossover([1, 2, 4], [2.0, 1.8, 1.5]).status == "no_crossover_in_range"
    assert analysis.find_crossover([1], [2.0]).status == "insufficient"


def test_find_crossover_exactly_one_is_not_a_crossing_until_it_drops():
    co = analysis.find_crossover([1, 2, 4], [1.5, 1.0, 0.8])
    assert co.status == "crosses" and 2 <= co.c_star <= 4


def _row(kv, spec, ctx, c, tpot, **kw):
    return {"status": "ok", "kv_dtype": kv, "spec": spec, "prefix_caching": False,
            "workload": "w", "context_len": ctx, "concurrency": c, "rep": 0,
            "tpot_ms_p50": tpot, "throughput_tps": 1000.0 * c / tpot, **kw}


def test_aggregate_takes_median_over_reps():
    rows = [dict(_row("auto", "none", 128, 4, t), rep=i) for i, t in enumerate([10, 12, 100])]
    agg = analysis.aggregate(rows)
    assert len(agg) == 1 and agg[0]["tpot_ms_p50"] == 12 and agg[0]["n_reps"] == 3


def test_speedup_and_crossover_end_to_end():
    # base TPOT flat 20ms; spec TPOT rises 10 -> 40ms, so S = 2,1.33,1,0.5
    cs = [1, 4, 16, 64]
    spec_tpot = [10.0, 15.0, 20.0, 40.0]
    rows = [_row("auto", "none", 128, c, 20.0) for c in cs]
    rows += [_row("auto", "ngram", 128, c, t) for c, t in zip(cs, spec_tpot)]
    agg = analysis.aggregate(rows)
    c, s = analysis.speedup_series(agg, "auto", "ngram", 128, "w")
    assert c == cs and s == pytest.approx([2.0, 20 / 15, 1.0, 0.5])
    table = analysis.crossover_table(agg)
    assert len(table) == 1 and table[0]["status"] == "crosses"
    assert 16 <= table[0]["c_star"] <= 64


def test_capacity_gain_and_pressure_onset():
    rows = []
    for kv, blocks in (("auto", 1000), ("fp8", 2000)):
        analytic = blocks * 16 // 544
        for c in (8, 32, 64):
            pressured = c > analytic
            rows.append(_row(kv, "none", 512, c, 20.0, kv_blocks=blocks, block_size=16,
                             max_concurrency_analytic=analytic,
                             srv_preemptions_delta=5 if pressured else 0))
    cap = analysis.capacity_table(analysis.aggregate(rows))
    gain = analysis.capacity_gain(cap)
    assert gain[0]["analytic_gain_x"] == pytest.approx(2.0, rel=0.05)
    by = {r["kv_dtype"]: r for r in cap}
    assert by["auto"]["first_pressure_C"] == 32 and by["fp8"]["first_pressure_C"] == 64


def test_throughput_knee_detects_plateau():
    series = [{"concurrency": c, "throughput_tps": t}
              for c, t in [(1, 100), (8, 700), (32, 1000), (128, 1010), (256, 1000)]]
    k = analysis.throughput_knee(series)
    assert k["peak_tps"] == 1010 and k["knee_C"] == 32


def test_quality_deltas():
    q = [{"kind": "perplexity", "kv_dtype": "auto", "context_len": 512, "ppl": 5.00},
         {"kind": "perplexity", "kv_dtype": "fp8", "context_len": 512, "ppl": 5.05}]
    d = analysis.quality_deltas(q)
    assert d[0]["delta"] == pytest.approx(0.05) and d[0]["delta_pct"] == pytest.approx(1.0)
