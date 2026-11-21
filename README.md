# Balancing Memory and Compute in LLM Serving

**Empirical Research & Benchmark Harness**: FP8 KV Cache Precision vs. Speculative Decoding Across Load  
**Author**: Mridul Pant (`cs24mtech15009`), Department of Computer Science & Engineering, IIT Hyderabad  
**Target Model**: `meta-llama/Meta-Llama-3-8B-Instruct`  
**Serving Framework**: vLLM v0.32.0 (V1 Engine Core, FlashInfer Attention Backend, PagedAttention block size $= 16$)  
**Benchmark Hardware**: NVIDIA RTX PRO 5000 Blackwell ($48.9\text{ GB}$ GDDR7 VRAM, `sm120`)

---

## 1. Project Overview

Serving Large Language Models (LLMs) requires balancing two distinct physical hardware boundaries:
1. **The Memory Capacity & Bandwidth Wall (KV Cache)**: During autoregressive token generation, retaining Key-Value activations across hundreds of concurrent requests consumes gigabytes of high-bandwidth memory (HBM). Once VRAM fills, request queueing explodes and thrashing occurs due to memory preemptions.
2. **The Compute Wall (Batching & Verification FLOPs)**: Speculative decoding accelerates per-token generation under low concurrency ($C \le 4$), but requires the target model to verify speculative candidates. At high batch concurrency ($C \ge 64$), Tensor Cores saturate, causing speculative verification overhead to exceed baseline decoding time.

This project delivers an end-to-end empirical benchmark suite and theoretical framework that measures how **FP8 KV cache quantization** interacts with **speculative decoding** (N-gram prompt lookup, EAGLE, and a draft model) across varying sequence lengths and concurrency levels.

---

## 2. Key Empirical Findings

Llama-3-8B-Instruct on one RTX PRO 5000 Blackwell (vLLM 0.32.0). 221 measured rows: Stage 1 (60 rows) and the FP16 N-gram sweep (20 rows) from the first run, 141 rows from a clean re-run on 21 November 2026 (3 repeats). Every number is recomputed by `evaluation/make_analysis.py` from `results_final/results.jsonl`. Details and caveats: [`docs/Project_Report.md`](docs/Project_Report.md).

1. **KV capacity**: FP8 grows the KV pool from 13,249 to 26,099 blocks (**1.97x**); the re-run reproduces it (13,246 to 26,092).
2. **FP8 under load (C=256)**: throughput +45% at ctx 2,048 (782 to 1,137 tok/s) and +38% at ctx 7,680 (219 to 302). Preemptions fall from 35 to 9 at ctx 2,048 and from 19 to 0 at ctx 7,680. Mean request latency falls 31% and 27%. Per-token latency (TPOT p50) rises 30% and 48%, because the batch is larger.
3. **Speculation pays only at the lowest loads, and FP8 does not rescue it**: with FP16 the draft model gives 1.50x at C=1 and 1.35x at C=4, parity at C=16, and every method loses from C=64. With FP8 the draft model gives only 1.03x at C=1. From C=16 the best configuration is FP8 without speculation (603, 1,009 and 1,122 tok/s at C=16, 64 and 256). No interpolated crossover is quoted (five concurrency levels).
4. **Speculator KV tax**: the draft model takes 27% of the KV pool, EAGLE 10%, N-gram 0.3%.
5. **Quality**: FP8 raises WikiText-2 perplexity by at most 0.26% (contexts 512 to 4,096). HumanEval: 98/164 (FP16) versus 96/164 (FP8), within noise at this sample size. FP16 failed at ctx 7,680, so that tier has no pair. **MMLU was not run.**
6. **Prefix caching (cache-off control run)**: at C=256 the cache adds +63% (FP16: 766 to 1,247 tok/s) and +68% (FP8: 1,124 to 1,888). FP8 over FP16 is +47% with the cache off and +51% with it on, so most of FP8's gain is capacity.

**Known limits:** 3 repeats in the re-run and 2 in Stage 1 and FP16 N-gram, 5 concurrency levels, two rented instances (FP16 baselines agree within 2%), no MMLU, no finer grid, one context length for speculation. The first run's `results.jsonl` was truncated and 64 rows were rebuilt from `run.log` with placeholder telemetry; they are **not used** (the re-run replaced them; 29 of 32 comparable cells agree within 5%, but the FP8 draft model at C=1, 4, 16 was overstated by 10-19%). They stay in `results_budget/` for transparency. Most code was written with AI assistance; see the report, section 10.

---

## 3. Publication Figures

| Figure | Description | Location |
|---|---|---|
| **Figure 1** | Throughput Knees & Thrashing Collapse ($\text{ctx} \in \{512, 2048, 7680\}$) | `results_budget/report/figure1_kv_throughput_knee.png` |
| **Figure 2** | Speculative Crossover ($C^*$) Dual-Panel ($S_{\text{TPOT}}$ vs. $S_{\text{TPS}}$) | `results_budget/report/figure2_speculative_crossover.png` |
| **Figure 3** | PagedAttention Concurrency Limits & Speculator KV Block Tax | `results_budget/report/figure3_kv_capacity_gain.png` |
| **Figure 4** | Quality (WikiText-2 Perplexity + HumanEval) & Prefix Caching Synergy | `results_budget/report/figure4_quality_and_prefix_synergy.png` |

The figures used in the report and the slides are `evaluation/output/01_*.png` to `11_*.png` (and `docs/figures/`). Figures 1 to 4 above are older draft figures from the first run; where they disagree with the report, the report is right.

---

## 4. Repository Structure

```
llm_serving/
├── docs/
│   ├── Project_Report.md              # Full 10-section final project report
│   └── Architecture_Decision_Records.md # ADR log (including mandatory 9 Oct record)
├── configs/
│   ├── budget.yaml          # Standard $5 budget benchmark matrix (12 stages)
│   ├── experiments.yaml     # Full production matrix across multi-repetition sweeps
│   ├── extra.yaml           # Fine-grained follow-up matrix
│   └── pilot.yaml           # Fast sanity-check config (1-2 minutes)
├── src/llm_serving/
│   ├── __init__.py
│   ├── cli.py               # CLI entrypoint (`llm-serving run`, `quality`, `analyze`)
│   ├── config.py            # Pydantic v2 configuration models and run expander
│   ├── load_client.py       # Asynchronous closed-loop load generator (TTFT, TPOT, E2E)
│   ├── mock_server.py       # Deterministic CPU mock server for testing without GPU
│   ├── orchestrator.py      # Experiment runner, server lifecycle manager & resume logic
│   ├── quality_eval.py      # Perplexity and HumanEval evaluation via vLLM
│   ├── server.py            # vLLM background API server lifecycle and log scraper
│   ├── telemetry.py         # Prometheus metrics scraping & NVML GPU telemetry
│   ├── workload.py          # Synthetic, WikiText-2 corpus, and Shared-Prefix workloads
│   └── analysis.py          # Data aggregation, C* crossover finder, and figure plotter
├── scripts/
│   ├── generate_publication_charts.py  # Standalone 300-DPI publication chart generator
│   ├── run_humaneval.py               # Automated HumanEval 164-task evaluation script
│   ├── collect_remaining.sh           # Step-by-step orchestrator for follow-up runs
│   ├── prepare_corpus.py              # Auto-downloads and cleans WikiText-2 test set
│   ├── run_experiments.sh             # Master execution script for Vast.ai / RunPod
│   └── setup_server.sh                # System dependencies, PyTorch, vLLM installer
├── tests/
│   ├── test_logic.py        # 20 unit tests for formulas, crossover math & config validation
│   └── test_pipeline_mock.py# 4 end-to-end integration tests using MockServer
├── results_budget/          # First run (156 rows, 64 of them rebuilt and not used), quality logs
│   ├── results.jsonl        # Raw line-by-line benchmark measurements
│   ├── quality.jsonl        # Raw perplexity and HumanEval evaluations
│   ├── lm_eval/             # Raw HumanEval task execution JSON outputs
│   └── report/              # Older generated summary and figures (superseded by evaluation/output)
├── results_extra/           # Clean re-run, 21 Nov 2026 (rows, run.log, server_logs/, env.txt, pip_freeze.txt)
├── results_final/           # Merged analysis dataset built by evaluation/merge_results.py (+ rebuilt_vs_rerun.csv)
├── evaluation/              # merge_results.py, make_analysis.py, make_regime_map.py, output/
├── .env.example             # Safe template for environment variables (HF_TOKEN)
├── pyproject.toml           # Packaging, dependencies, and CLI entrypoint definition
└── README.md
```

---

## 5. Getting Started & Reproducibility

### 5.1 Installation

```bash
# Clone the repository
git clone https://github.com/mridulpant2010/llm_serving.git
cd llm_serving

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### 5.2 Running the Test Suite (CPU Only, No GPU Needed)

```bash
# Run unit tests
pytest tests/test_logic.py

# Run end-to-end pipeline integration tests
pytest tests/test_pipeline_mock.py
```

### 5.3 Executing Benchmarks on a GPU Instance (Vast.ai / RunPod)

```bash
# Setup environment and dependencies
bash scripts/setup_server.sh

# Run the budget matrix (12 server launches) inside tmux
bash scripts/run_experiments.sh budget

# Run WikiText-2 quality evaluation
llm-serving quality -c configs/budget.yaml --skip-lm-eval

# Re-run of Stages 3-5 (3 repeats, cache-off control) on a second instance
bash scripts/collect_remaining.sh rerun && bash scripts/collect_remaining.sh prefix

# Generate the older summary report and charts
llm-serving analyze --results results_budget/results.jsonl \
                    --quality results_budget/quality.jsonl \
                    --out results_budget/report
```

### 5.4 Re-generating Figures and Tables

```bash
python evaluation/merge_results.py results_budget results_extra results_final   # merged dataset
python evaluation/make_analysis.py results_final evaluation/output      # charts 01-10 + comparison_tables.xlsx
python evaluation/make_regime_map.py results_final evaluation/output    # chart 11 (regime map)
python scripts/generate_publication_charts.py                            # older figures 1-4
```

---

## 6. Official Submission & Defense Artifacts

Detailed academic analysis, mathematical derivations, decision records, and final reports:
- **Project Report**: [`docs/Project_Report.md`](docs/Project_Report.md)
- **Architecture Decision Records (ADRs)**: [`docs/Architecture_Decision_Records.md`](docs/Architecture_Decision_Records.md)
- **Evaluation scripts**: [`evaluation/`](evaluation/README.md)
- **Environment template**: `.env.example` (never commit real tokens; the `vastai` key must not be tracked)

---

## 7. Historical planning documents

The Markdown files in the repository root (`project_report.md`, `project_report_t2_t3.md`, `final_project_report.md`, `progress_report_design.md`, `experiment_coding_progress_report.md`, `project_analysis.md`, `subtopic_selection.md`, `system_design.md`, `system_design_first_principles.md`) are the planning and progress documents written during the project. They are kept because they are the evidence for the original plan (see ADR-001). **They are not the final results: where they disagree with `docs/Project_Report.md` or `docs/Architecture_Decision_Records.md`, the files in `docs/` are correct.**

## 8. License

MIT, see [`LICENSE`](LICENSE). Third-party models and datasets keep their own licences.
