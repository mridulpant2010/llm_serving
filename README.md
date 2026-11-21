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

Llama-3-8B-Instruct on one RTX PRO 5000 Blackwell (vLLM 0.32.0), 152 successful runs, 2 repeats. Every number is recomputed from `results_budget/results.jsonl` by `evaluation/make_analysis.py`. Details and caveats: [`docs/Project_Report.md`](docs/Project_Report.md).

1. **KV capacity**: FP8 grows the KV pool from 13,249 to 26,099 blocks (**1.97x**).
2. **FP8 under load (C=256)**: throughput +45% at ctx 2,048 (782 to 1,137 tok/s) and +38% at ctx 7,680 (219 to 302). Preemptions fall from 35 to 9 at ctx 2,048 and from 19 to 0 at ctx 7,680. Mean request latency falls 31% and 27%. Per-token latency (TPOT p50) rises 30% and 48%, because the batch is larger. Peak to peak at ctx 2,048 the gain is +33%.
3. **Speculation pays only at low concurrency**: every method helps at C=1 and 4, the draft model to about C=16, and all lose from C=64. From C=64 the best configuration is FP8 without speculation. This is read from five measured concurrency levels, so no interpolated crossover is quoted.
4. **Speculator KV tax**: the draft model takes 27% of the KV pool, EAGLE 10%, N-gram 0.3%.
5. **Quality**: FP8 raises WikiText-2 perplexity by at most 0.26% (contexts 512 to 4,096). HumanEval: 98/164 (FP16) versus 96/164 (FP8), within noise at this sample size. FP16 failed at ctx 7,680, so that tier has no pair. **MMLU was not run.**
6. **Prefix caching**: with the cache on, FP8 reaches 1,906 tok/s versus 1,264 for FP16 (+51%). The cache's own effect is **not measured** (no cache-off control).

**Known limits:** 64 of 156 rows (Stage 3 draft model C=256 and EAGLE, all of Stages 4 and 5) were rebuilt from `run.log` after a truncated download, with placeholder telemetry (only throughput, TPOT, KV blocks, preemptions and acceptance are used), 2 repeats, 5 concurrency levels. Most code was written with AI assistance; see the report, section 10. Follow-up runs are specified in `configs/extra.yaml` and `scripts/collect_remaining.sh`.

---

## 3. Publication Figures

| Figure | Description | Location |
|---|---|---|
| **Figure 1** | Throughput Knees & Thrashing Collapse ($\text{ctx} \in \{512, 2048, 7680\}$) | `results_budget/report/figure1_kv_throughput_knee.png` |
| **Figure 2** | Speculative Crossover ($C^*$) Dual-Panel ($S_{\text{TPOT}}$ vs. $S_{\text{TPS}}$) | `results_budget/report/figure2_speculative_crossover.png` |
| **Figure 3** | PagedAttention Concurrency Limits & Speculator KV Block Tax | `results_budget/report/figure3_kv_capacity_gain.png` |
| **Figure 4** | Quality (WikiText-2 Perplexity + HumanEval) & Prefix Caching Synergy | `results_budget/report/figure4_quality_and_prefix_synergy.png` |

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
├── results_budget/          # Complete empirical dataset (156 runs, quality logs & report)
│   ├── results.jsonl        # Raw line-by-line benchmark measurements
│   ├── quality.jsonl        # Raw perplexity and HumanEval evaluations
│   ├── lm_eval/             # Raw HumanEval task execution JSON outputs
│   └── report/              # Generated markdown summary, json, and figures
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

# Generate summary report and charts
llm-serving analyze --results results_budget/results.jsonl \
                    --quality results_budget/quality.jsonl \
                    --out results_budget/report
```

### 5.4 Re-generating Figures and Tables

```bash
python evaluation/make_analysis.py results_budget evaluation/output      # charts 01-10 + comparison_tables.xlsx
python evaluation/make_regime_map.py results_budget evaluation/output    # chart 11 (regime map)
python scripts/generate_publication_charts.py                            # older figures 1-4
```

---

## 6. Official Submission & Defense Artifacts

Detailed academic analysis, mathematical derivations, decision records, and final reports:
- **Project Report**: [`docs/Project_Report.md`](docs/Project_Report.md)
- **Architecture Decision Records (ADRs)**: [`docs/Architecture_Decision_Records.md`](docs/Architecture_Decision_Records.md)
- **Evaluation scripts**: [`evaluation/`](evaluation/README.md)
- **Environment template**: `.env.example` (never commit real tokens; the `vastai` key must not be tracked)
