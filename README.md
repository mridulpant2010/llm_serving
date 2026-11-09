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

This project delivers an end-to-end empirical benchmark suite and theoretical framework that explores the **Pareto frontier** between **FP8 KV cache quantization** and **speculative decoding architectures** (Prompt Lookup / N-gram, EAGLE-2, and Draft Models) across varying sequence lengths and concurrency levels.

---

## 2. Key Empirical Findings

1. **$1.97\times$ Physical KV Pool Doubling**:
   - FP16 allocates $13,249$ PagedAttention blocks ($211,984$ tokens).
   - FP8 (`e4m3`) allocates **$26,099$ blocks ($417,584$ tokens)**.
   - Doubles maximum analytic concurrency capacity across all context tiers ($512, 2048, 7680$ tokens).
2. **Thrashing Elimination & Knee Shift**:
   - At $\text{ctx}=2048, C=256$, FP16 suffers **$35$ request preemptions**, collapsing throughput by $-8.8\%$ ($857.2 \rightarrow 782.0\text{ tok/s}$) with extreme tail latency ($\text{P99 TTFT} = 73.5\text{ s}$).
   - FP8 sustains **$1,136.5\text{ tok/s}$ ($+45.3\%$ peak TPS boost)** with **zero preemptions**.
   - At long context ($\text{ctx}=7680$), FP8 shifts the saturation knee $4\times$ deeper into concurrency ($C=16 \rightarrow C=64$).
3. **Discovery of the Speculative Crossover Point ($C^*$)**:
   - Speculative decoding is not universally beneficial:
     - **N-gram / Prompt Lookup**: High load triggers a **$-17.1\%$ throughput penalty** at $C=256$ due to low acceptance ($\alpha=11.4\%$), crossing into slowdown at **$C^* = 25.9$**.
     - **EAGLE Head**: Crosses into slowdown at **$C^* = 16.9$** under FP16.
     - **Draft Model (Llama-3.2-1B)**: High acceptance ($\alpha=52.5\%$) amortizes verification, sustaining $1.44\times - 2.15\times$ speedup across all tested loads.
4. **Interaction Effect (FP8 Shifts $C^*$ Outward)**:
   - Halving KV cache bandwidth during attention verification expands the speculative operating window:
   - EAGLE crossover point shifts from **$C^* = 16.9 \rightarrow 43.9$ ($+2.60\times$)**!
   - N-gram crossover penalty is completely eliminated inside the tested concurrency range.
5. **Zero-Degradation Quality Verification**:
   - Evaluated on WikiText-2 across $491,456$ tokens.
   - Relative perplexity degradation is strictly bounded under **$0.26\%$** across context lengths ($512$ to $7680$ tokens).
6. **Prefix Caching & Radix Attention Synergy**:
   - Combining FP8 KV cache with Prefix Caching pushes serving throughput to **$1,905.7\text{ tok/s}$** ($+144.4\%$ over vanilla FP16).

---

## 3. Publication Figures

| Figure | Description | Location |
|---|---|---|
| **Figure 1** | Throughput Knees & Thrashing Collapse ($\text{ctx} \in \{512, 2048, 7680\}$) | `results_budget/report/figure1_kv_throughput_knee.png` |
| **Figure 2** | Speculative Crossover ($C^*$) Dual-Panel ($S_{\text{TPOT}}$ vs. $S_{\text{TPS}}$) | `results_budget/report/figure2_speculative_crossover.png` |
| **Figure 3** | PagedAttention Concurrency Limits & Speculator KV Block Tax | `results_budget/report/figure3_kv_capacity_gain.png` |
| **Figure 4** | Quality (WikiText-2 Perplexity) & Prefix Caching Synergy | `results_budget/report/figure4_quality_and_prefix_synergy.png` |

---

## 4. Repository Structure

```
llm_serving/
├── configs/
│   ├── budget.yaml          # Standard $5 budget benchmark matrix (12 stages)
│   ├── experiments.yaml     # Full production matrix across multi-repetition sweeps
│   └── pilot.yaml           # Fast sanity-check config (1-2 minutes)
├── src/llm_serving/
│   ├── __init__.py
│   ├── cli.py               # CLI entrypoint (`llm-serving run`, `quality`, `analyze`)
│   ├── config.py            # Pydantic v2 configuration models and run expander
│   ├── load_client.py       # Asynchronous closed-loop load generator (TTFT, TPOT, E2E)
│   ├── mock_server.py       # Deterministic CPU mock server for testing without GPU
│   ├── orchestrator.py      # Experiment runner, server lifecycle manager & resume logic
│   ├── quality_eval.py      # WikiText-2 Perplexity evaluation via /v1/completions
│   ├── server.py            # vLLM background API server lifecycle and log scraper
│   ├── telemetry.py         # Prometheus metrics scraping & NVML GPU telemetry
│   ├── workload.py          # Synthetic, WikiText-2 corpus, and Shared-Prefix workloads
│   └── analysis.py          # Data aggregation, C* crossover finder, and figure plotter
├── scripts/
│   ├── generate_publication_charts.py  # Standalone 300-DPI publication chart generator
│   ├── prepare_corpus.py              # Auto-downloads and cleans WikiText-2 test set
│   ├── run_experiments.sh             # Master execution script for Vast.ai / RunPod
│   └── setup_server.sh                # System dependencies, PyTorch, vLLM installer
├── tests/
│   ├── test_logic.py        # 20 unit tests for formulas, crossover math & config validation
│   └── test_pipeline_mock.py# 4 end-to-end integration tests using MockServer
├── results_budget/          # Complete empirical dataset (156 runs, quality logs & report)
│   ├── results.jsonl        # Raw line-by-line benchmark measurements
│   ├── quality.jsonl        # Raw perplexity evaluations
│   └── report/              # Generated markdown summary, json, and figures
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

# Run the 12-stage budget matrix inside tmux
bash scripts/run_experiments.sh budget

# Run WikiText-2 quality evaluation
llm-serving quality -c configs/budget.yaml --skip-lm-eval

# Generate summary report and charts
llm-serving analyze --results results_budget/results.jsonl \
                    --quality results_budget/quality.jsonl \
                    --out results_budget/report
```

### 5.4 Re-generating Publication Figures Locally

```bash
python scripts/generate_publication_charts.py
```

---

## 6. Citation & Defense Artifacts

Detailed academic analysis, mathematical derivations, and defense slide decks are located in:
- [`empirical_experiment_analysis.md`](empirical_experiment_analysis.md)
- `Balancing Memory and Compute in LLM Serving.pptx`
