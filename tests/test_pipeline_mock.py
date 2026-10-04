"""End-to-end test of the harness against the MOCK server (no GPU needed).

This verifies the plumbing: launching, load generation, streaming timestamps,
metric scraping, JSONL output, resume, perplexity-through-server and the
analysis pipeline.  It does NOT validate anything about real vLLM behaviour.
"""

import json

import pytest

from llm_serving import analysis, orchestrator, quality_eval
from llm_serving.config import ExperimentConfig, FixedSettings, expand_runs
from llm_serving.load_client import run_closed_loop, summarize
from llm_serving.mock_server import MockServer
from llm_serving.config import ServerRun
from llm_serving.workload import SyntheticWorkload


def _run(kv="auto", spec="none"):
    return ServerRun(
        stage="t", kv_dtype=kv, spec_name=spec,
        spec_config=None if spec == "none" else {"num_speculative_tokens": 4},
        prefix_caching=False, workload="synthetic", workload_args={},
        context_lengths=(64,), concurrency=(1,), max_model_len=512, max_num_seqs=64)


def test_load_client_measures_ttft_tpot_and_counts_tokens_from_usage():
    import asyncio

    with MockServer(_run(spec="eagle"), time_scale=0.2) as srv:
        reqs = SyntheticWorkload().make_requests(6, 64, 24, seed=0)
        load = asyncio.run(run_closed_loop(srv.base_url, "mock", reqs, concurrency=3))
    row = summarize(load)
    assert row["num_failed"] == 0
    assert row["output_tokens_total"] == 6 * 24        # from usage, though chunks carry >1 token
    assert row["ttft_ms_p50"] > 0 and row["tpot_ms_p50"] > 0
    assert row["throughput_tps"] > 0


def test_fp8_doubles_reported_kv_blocks():
    from llm_serving.telemetry import get_cache_info

    with MockServer(_run("auto")) as a, MockServer(_run("fp8")) as b:
        ba, bb = get_cache_info(a.base_url), get_cache_info(b.base_url)
    assert bb["num_gpu_blocks"] == 2 * ba["num_gpu_blocks"]


def test_perplexity_through_server_sees_fp8_penalty():
    ids = list(range(1000, 1200))
    windows = [ids, ids[::-1]]
    with MockServer(_run("auto")) as a, MockServer(_run("fp8")) as b:
        p0 = quality_eval.perplexity_via_server(a.base_url, "mock", windows)
        p1 = quality_eval.perplexity_via_server(b.base_url, "mock", windows)
    assert p1["ppl"] > p0["ppl"] and p0["n_failed"] == 0


@pytest.mark.slow
def test_full_pipeline_finds_a_crossover_and_resumes(tmp_path):
    cfg = ExperimentConfig(
        model="mock",
        fixed=FixedSettings(output_len=24, repetitions=1, rounds_per_level=3,
                            min_requests_per_level=6, warmup_requests=4, max_num_seqs=128),
        speculation={"none": None,
                     "ngram": {"method": "ngram", "num_speculative_tokens": 4},
                     "eagle": {"method": "eagle", "num_speculative_tokens": 4}},
        stages={"s": {"workload": "synthetic", "kv_dtypes": ["auto", "fp8"],
                      "speculation": ["none", "eagle"], "context_lengths": [64],
                      "concurrency": [1, 4, 16, 64, 128]}},
    )
    out = tmp_path / "mock_results.jsonl"
    orchestrator.run_experiments(cfg, expand_runs(cfg), "mock", out, log=lambda m: None)

    rows = analysis.load_results(out)
    assert len(rows) == 4 * 5                       # 2 kv x 2 spec x 5 concurrency levels
    assert all(r["backend"] == "mock" for r in rows)

    # Resume must not duplicate measurements
    orchestrator.run_experiments(cfg, expand_runs(cfg), "mock", out, log=lambda m: None)
    assert len(analysis.load_results(out)) == len(rows)

    agg = analysis.aggregate(rows)
    table = {(r["kv_dtype"], r["spec"]): r for r in analysis.crossover_table(agg)}
    r = table[("auto", "eagle")]
    assert r["speedup_at_min_C"] > 1.2            # speculation helps at C=1 in the toy model
    assert r["speedup_at_max_C"] < 1.0            # and hurts at C=128
    assert r["status"] == "crosses"

    summary = analysis.generate_report(out, None, tmp_path / "report")
    assert "mock" in summary["backends"]
    md = (tmp_path / "report" / "summary.md").read_text()
    assert "MOCK server" in md                    # banner so mock data can't be misread
    assert (tmp_path / "report" / "cstar_shift.png").exists()
