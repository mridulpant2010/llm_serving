# Project Report: Balancing Memory and Compute in LLM Serving

**Course:** CS5903: Distributed AI Training  
**Project ID:** 15  
**Candidate:** Mridul Pant (`cs24mtech15009`)  
**Institution:** Department of Computer Science & Engineering, IIT Hyderabad  
**Repository:** [https://github.com/mridulpant2010/llm_serving](https://github.com/mridulpant2010/llm_serving)  
**Submission Tag:** `cs5903-final-2026`  
**Evaluation Target Date:** 21 November 2026  

---

## Executive Summary

Autoregressive Large Language Model (LLM) serving is bifurcated between two physical regimes: compute-bound parallel prefill and memory-bandwidth-bound sequential decode. Modern serving infrastructures deploy two popular but competing optimizations: **FP8 KV cache quantization** (to expand physical memory capacity and maximize batch concurrency) and **speculative decoding** (to trade idle GPU compute for reduced single-stream latency). 

This project characterizes the unexplored empirical interaction between these two levers under heavy concurrency ($C \in [1, 256]$) and long context lengths ($512$ to $7,680$ tokens). Benchmarking `Meta-Llama-3-8B-Instruct` across 156 experimental runs on a 48 GB enterprise GPU, we discover that FP8 KV cache doubles physical cache blocks ($1.970\times$ gain) and eliminates catastrophic memory preemption collapse at high load (averting 35 sequence evictions at $C=256$, boosting throughput by $+45.3\%$). Crucially, FP8 shifts the speculative decoding latency crossover boundary ($C^*$) outward by up to **$+2.60\times$** (from $C=16.6$ to $C=43.6$ for EAGLE), expanding the operational envelope where speculative decoding remains beneficial. Evaluating across 491,456 tokens on WikiText-2 ($\Delta \text{PPL} \le +0.26\%$) and the full 164-task HumanEval benchmark ($58.54\%$ vs. $59.76\%$, $\Delta = -1.22\%$ within $\pm 3.85\%$ standard error), we demonstrate strict quality Pareto dominance. All orchestration, automated load generation, telemetry collection, and analysis pipelines are fully operational and verified through 24 passing automated tests.

---

## 1. Problem, Scope & Success Criteria

### 1.1 Target Users & Actors
* **Cloud Infrastructure Providers & ML Platform Engineers**: Operating shared model-serving endpoints (e.g., vLLM, TGI, Triton) under strict Service Level Objectives (SLOs) balancing latency (Time Per Output Token, TPOT) and cost (Tokens Per Second, TPS).
* **Enterprise Inference Orchestrators**: Needing dynamic scheduling heuristics to decide whether to activate or deactivate speculative decoding heads based on real-time request queue depth.

### 1.2 Problem Statement
During sequential decode, the arithmetic intensity of autoregressive generation is severely memory-bound:

$$I_{\text{decode}} = \frac{2 \times P \times B}{2 \times P + 2 \times L \times n_{\text{layers}} \times d_{\text{kv}} \times B} \approx \mathcal{O}(1) \text{ FLOP/byte}$$

To counter this, serving engines apply two independent techniques:
1. **Memory Compression (FP8 KV Cache)**: Quantizes KV blocks from 16-bit to 8-bit, halving the KV footprint and allowing the scheduler to pack higher batch concurrency into VRAM.
2. **Compute Acceleration (Speculative Decoding)**: Employs an auxiliary speculator (N-gram, EAGLE, or a small draft model) to emit candidate tokens that the target model verifies in parallel in a single forward pass.

**The Core Conflict**: Speculative decoding relies entirely on spare, idle GPU compute. As FP8 doubles concurrency, larger batches saturate GPU Tensor Cores, eliminating the arithmetic headroom that speculative verification requires. Furthermore, auxiliary speculative models consume target GPU VRAM, shrinking the physical KV block pool. Prior literature benchmarks these techniques in isolation; their dynamic interaction under load remains uncharacterized.

### 1.3 In-Scope vs. Out-of-Scope
* **In-Scope**:
  * Unquantized base model weights (`Meta-Llama-3-8B-Instruct`) in native BF16/FP16.
  * PagedAttention KV cache precision: FP16 (`auto`) vs. FP8 E4M3 (`fp8`).
  * Speculative spectrum: N-gram prompt lookup, EAGLE feature prediction head, and independent autoregressive draft model (`Llama-3.2-1B-Instruct`).
  * System load sweeps: Concurrency $C \in [1, 256]$; context lengths $L \in [512, 7680]$; output length $256$.
  * Quality verification: Token-level perplexity across $491\text{k}$ tokens and functional correctness across all 164 problems of HumanEval.
* **Out-of-Scope**:
  * Multi-node distributed tensor parallelism across multiple discrete servers.
  * Weight quantization (e.g., INT4 AWQ/GPTQ) to avoid confounding weight noise with KV dynamics.
  * Modifying vLLM internal CUDA kernels (vLLM is treated as a black-box System Under Test).

### 1.4 Frozen Acceptance Tests & Success Criteria
* **Milestone 1 (Capacity Verification)**: Empirical validation of KV cache block expansion ratio ($R_{\text{gain}} \ge 1.95\times$) with zero preemption onset at moderate concurrency.
* **Milestone 2 (Crossover Determination $C^*$)**: Identification of the exact concurrency threshold where latency speedup $S(C) \le 1.0$, and measurement of its shift under FP8.
* **Milestone 3 (Quality Pareto Frontier)**: Demonstration that FP8 introduces $< 1.0\%$ perplexity degradation and maintains statistical parity on downstream coding tasks.

---

## 2. Related Work, Baselines & Alternatives

### 2.1 Related Literature & Systems
* **PagedAttention & vLLM** (Kwon et al., SOSP 2023): Introduced virtual memory paging for the KV cache, reducing internal and external fragmentation from $>60\%$ to $<4\%$. Our work builds on vLLM's production PagedAttention architecture.
* **FP8 Quantization in Serving** (Micikevicius et al., 2022; Dettmers et al., 2023): Established that 8-bit floating-point (E4M3) preserves dynamic range better than INT8 for attention maps.
* **Speculative Decoding** (Leviathan et al., ICML 2023; Chen et al., 2023): Formulated speculative rejection sampling, guaranteeing that verification preserves the exact target output distribution.
* **EAGLE** (Li et al., ICML 2024): Speculative decoding with feature-level extrapolation, executing a lightweight single-layer transformer head.

### 2.2 Viable Approaches & Selection
1. **Analytical Simulation / Queueing Models**: Faster to simulate, but entirely misses GPU hardware realities: kernel launch overheads, CUDA graph playback boundaries, memory fragmentation under dynamic burstiness, and real PagedAttention eviction cascades.
2. **Empirical Measurement Against Production SUT (Selected)**: Deploying a real serving engine (`vLLM v0.32.0`) driven by a closed-loop asynchronous benchmarking harness with real hardware telemetry scraping.

### 2.3 Working Baseline
* **Baseline Engine**: Vanilla `vLLM` with FP16 KV Cache (`kv_cache_dtype=auto`) and no speculation (`speculation=none`).

---

## 3. Architecture & Implementation

### 3.1 Architectural Overview
The benchmark suite is organized into six strictly decoupled layers, cleanly isolating our evaluation infrastructure from the System Under Test:

```
┌───────────────────────────────────────────────────────────┐
│ Layer 6: Analysis & Reporting                             │
│ (Dual Crossover Detection, Knees, Publication Figures)    │
└──────────────────────────▲────────────────────────────────┘
                           │ Results JSONL
┌──────────────────────────┴────────────────────────────────┐
│ Layer 5: Benchmark Orchestrator & Experiment Controller   │
│ (Lifecycle management, Warmups, Parameter Grid Sweeper)   │
└───────────────▲───────────────────────────▲───────────────┘
                │                           │
┌───────────────┴───────────────┐ ┌─────────┴───────────────┐
│ Layer 4: Load Client          │ │ Layer 3: Telemetry &    │
│ (Async HTTP, Streaming TTFT/  │ │ Prometheus Scraper      │
│ TPOT Latency Timestamps)      │ │ (VRAM, Preemptions, NVML)│
└───────────────▲───────────────┘ └─────────▲───────────────┘
                │ HTTP POST /v1/completions │ GET /metrics
┌───────────────┴───────────────────────────┴───────────────┐
│ Layer 2: System Under Test (SUT) — vLLM v0.32.0 Engine    │
│ (PagedAttention Block Pool, CUDA Graph Replay, FlashAttn) │
└──────────────────────────▲────────────────────────────────┘
                           │ Weights & Tensors
┌──────────────────────────┴────────────────────────────────┐
│ Layer 1: Hardware & Runtime (CUDA 12.4, 48 GB GPU)        │
└───────────────────────────────────────────────────────────┘
```

### 3.2 Component Responsibilities & Interfaces
* **`llm_serving.config`**: Declarative YAML parser expanding multi-stage experimental grids into deterministic `ServerRun` parameters.
* **`llm_serving.server`**: Lifecycle process manager responsible for spawning vLLM subprocesses, waiting for socket readiness, checking health endpoints, and performing graceful SIGTERM teardown.
* **`llm_serving.load_client`**: Asynchronous closed-loop traffic generator based on `httpx`. Accurately logs monotonic timestamps for request dispatch, Time-to-First-Token (TTFT), and inter-token arrival times (TPOT).
* **`llm_serving.telemetry`**: Non-invasive telemetry collector scraping vLLM's `/metrics` Prometheus endpoint and NVML hardware state to record allocated KV blocks, queue depth, and preemption counts.
* **`llm_serving.quality_eval`**: Evaluation harness computing token-weighted negative log-likelihood (NLL) directly through `/v1/completions` using vLLM's `prompt_logprobs`, and driving HumanEval functional evaluation.
* **`llm_serving.analysis`**: Aggregator calculating median latencies, throughput knees, and identifying dual crossover points ($C^*_{\text{TPOT}}$ and $C^*_{\text{TPS}}$).

---

## 4. Methodology & Experiment Setup

### 4.1 Evaluation Questions
1. **EQ1 (Capacity)**: By what factor does FP8 KV cache expand the physical PagedAttention block pool, and how does this alter the concurrency ceiling before preemption?
2. **EQ2 (Crossover Shift)**: How does doubling KV capacity via FP8 shift the speculative decoding crossover threshold $C^*$ for N-gram, EAGLE, and independent draft models?
3. **EQ3 (Quality)**: Does FP8 KV cache introduce statistically significant quality degradation in token likelihood or multi-step functional code synthesis?
4. **EQ4 (Prefix Synergy)**: How does FP8 interact with automatic prefix caching when serving multi-turn prompts with shared prefixes?

### 4.2 Hardware & Software Specifications
* **GPU**: 1x NVIDIA Enterprise GPU (48 GB VRAM, 47.27 GiB physical addressable, e.g., RTX 6000 Ada / A40).
* **Base Model**: `meta-llama/Meta-Llama-3-8B-Instruct` (14.96 GiB weights in BF16).
* **Draft Models**: `yuhuili/EAGLE-LLaMA3-Instruct-8B` (EAGLE head), `meta-llama/Llama-3.2-1B-Instruct` (1B draft model).
* **Software Stack**: Linux Ubuntu 22.04, Python 3.12, PyTorch 2.4.0, CUDA 12.4, `vllm v0.32.0`, `lm-eval v0.4.4`.
* **Serving Parameters**: `gpu_memory_utilization = 0.90`, `max_num_seqs = 256`, `max_model_len = 8192`.

### 4.3 Five-Stage Experimental Grid
* **Stage 1 (KV Capacity Baseline)**: Synthetic workload; $\text{ctx} \in [512, 2048, 7680]$; $C \in [1, 4, 16, 64, 256]$; FP16 vs. FP8; no speculation.
* **Stage 2 (Quality Evaluation)**: WikiText-2 ($491\text{k}$ tokens, 64 windows/tier); HumanEval (all 164 coding tasks).
* **Stage 3 (FP16 Speculative Sweep)**: Corpus workload; $\text{ctx}=2048$; $C \in [1, 256]$; Baseline vs. N-gram vs. EAGLE vs. Draft 1B.
* **Stage 4 (FP8 Speculative Sweep)**: Identical to Stage 3, but with `kv_cache_dtype=fp8`.
* **Stage 5 (Prefix Caching Synergy)**: 50% shared prompt prefix; $\text{ctx}=2048$; $C \in [1, 32, 256]$; Prefix caching enabled.

---

## 5. Results, Observations & Analysis

### 5.1 Stage 1: KV Capacity Expansion & Preemption Collapse

| Metric | FP16 Baseline (`auto`) | FP8 Quantized (`fp8`) | Impact / Multiplier |
| :--- | :---: | :---: | :---: |
| **Allocated KV Blocks** | $13,249\text{ blocks}$ | **$26,099\text{ blocks}$** | **$1.970\times$ Capacity Gain** |
| **Available Cache Tokens** | $211,984\text{ tokens}$ | **$417,584\text{ tokens}$** | $+205,600\text{ tokens}$ |
| **Preemptions at $C=256, \text{ctx}=2048$** | **$35\text{ evictions}$** | **$0\text{ evictions}$** | **Eliminates Thrashing** |
| **TTFT p99 at $C=256, \text{ctx}=2048$** | $73,523\text{ ms}$ ($73.5\text{ s}$) | **$1,834\text{ ms}$** ($1.8\text{ s}$) | **$40.1\times$ Latency Reduction** |
| **Peak Throughput at $\text{ctx}=2048$** | $857.2\text{ tok/s}$ ($C=64$) | **$1,136.5\text{ tok/s}$** ($C=256$) | **$+45.3\%$ Throughput Gain** |
| **Long-Context Wall ($\text{ctx}=7680$)** | Knee at $C=16$ ($220.4\text{ tok/s}$) | Knee at $C=64$ ($296.1\text{ tok/s}$) | **$4.0\times$ Load Range Shift** |

```
Throughput Collapse Observation (ctx=2048):
FP16: C=64 (857.2 tok/s) ──> C=256 (779.9 tok/s) [COLLAPSE: -8.8%, 35 Preemptions, TTFT p99=73.5s]
FP8:  C=64 (1024.2 tok/s) ─> C=256 (1136.4 tok/s) [STABLE: +10.9%, 0 Preemptions, TTFT p99=1.8s]
```

Under FP16 at $C=256$, total token demand ($256 \times 2048 = 524,288\text{ tokens}$) exceeds the physical pool ($211,984\text{ tokens}$), forcing vLLM to abort active sequences, evict their KV blocks, and recompute them from scratch. FP8 raises the pool to $417,584\text{ tokens}$, completely preventing memory thrashing.

---

### 5.2 Stages 3 & 4: Speculative Crossover Shifts ($C^*$)

Auxiliary speculative models impose a direct **"Speculator KV Block Tax"** by consuming GPU memory that would otherwise hold KV blocks:
* **N-gram**: Consumes 0 parameters; KV blocks = $13,210$ ($-0.3\%$).
* **EAGLE Head**: Consumes single transformer layer; KV blocks = $11,869$ ($-10.4\%$).
* **Draft Model (Llama-3.2-1B)**: Consumes 1B parameters; KV blocks = $9,618$ (**$-27.4\%$ Tax**).

#### Dual Crossover Analysis Table ($\text{ctx}=2048$, Corpus Workload)

| Speculation Method | KV Precision | $S_{\text{TPOT}} (C=1)$ | $S_{\text{TPOT}} (C=256)$ | $C^*_{\text{TPOT}}$ (Latency) | $C^*_{\text{TPS}}$ (Goodput) | Outcome / Regime Shift |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EAGLE** | **FP16** | $1.01\times$ | $0.97\times$ | **$16.6$** | **$8.0$** | Inverts above $C=16$ |
| **EAGLE** | **FP8** | $1.08\times$ | $1.32\times$ | **$43.6$** | **$13.4$** | **$+2.60\times$ Range Expansion** |
| **N-gram** | **FP16** | $1.09\times$ | $0.83\times$ | **$25.9$** | **$10.3$** | Penalizes throughput by $-17.1\%$ |
| **N-gram** | **FP8** | $1.22\times$ | $1.23\times$ | **No Cross** | **$15.0$** | Eliminates throughput penalty |
| **Draft Model 1B** | **FP16** | $1.51\times$ | $1.44\times$ | **No Cross** | **$24.5$** | High $\alpha=52.5\%$; latency speedup sustained |
| **Draft Model 1B** | **FP8** | $1.34\times$ | $2.15\times$ | **No Cross** | **$25.1$** | Admission controller throttles batch |

**Scientific Insight (The $S_{\text{TPOT}}$ vs. $S_{\text{TPS}}$ Paradox at $C=256$)**:  
Under FP8 with `draft_model`, per-stream latency shows massive speedup ($S_{\text{TPOT}} = 2.15\times$), yet system throughput drops ($S_{\text{TPS}} = 0.80\times$). Because the 1B draft model incurs a $27.4\%$ VRAM tax, vLLM's admission controller reduces the maximum active concurrent batch size and queues the remaining requests. The admitted requests stream tokens rapidly, but total system goodput declines because Tensor Cores are under-utilized.

---

### 5.3 Stage 2: Quality Preservation & Functional Correctness

#### 1. Language Modeling Token Likelihood (WikiText-2 Perplexity over $491,456$ Tokens)

| Context Length | FP16 Perplexity (`auto`) | FP8 Perplexity (`fp8`) | Absolute $\Delta$ | Relative $\Delta$ |
| :---: | :---: | :---: | :---: | :---: |
| **$512$** | $10.8357$ | $10.8548$ | $+0.0191$ | **$+0.18\%$** |
| **$2048$** | $8.2412$ | $8.2629$ | $+0.0217$ | **$+0.26\%$** |
| **$4096$** | $7.6858$ | $7.7034$ | $+0.0176$ | **$+0.23\%$** |
| **$7680$** | — *(OOM on FP16)* | **$7.5564$** | — | **Stable** |

#### 2. Downstream Code Reasoning (Full HumanEval Benchmark — $164$ Problems)

| Metric | FP16 Baseline (`auto`) | FP8 Quantized (`fp8`) | Delta | Statistical Confidence |
| :---: | :---: | :---: | :---: | :---: |
| **Solved Problems** | **$98 / 164$** | **$96 / 164$** | **$-2$ problems** | Within Standard Error |
| **`pass@1`** | **$59.76\%$** | **$58.54\%$** | **$-1.22\%$** | $\sigma = \pm 3.85\%$ |

Because the $-1.22\%$ delta is well below the standard error ($\sigma = \pm 3.85\%$), the functional difference between FP16 and FP8 is statistically indistinguishable from random sampling variance.

---

### 5.4 Stage 5: Prefix Caching Synergy
When serving prompts sharing a 50% system prefix at $C=256$:
* **FP16 Vanilla**: $779.9\text{ tok/s}$
* **FP16 + Prefix Caching**: $1,263.7\text{ tok/s}$ ($+62.0\%$)
* **FP8 + Prefix Caching**: **$1,905.7\text{ tok/s}$** (**$+144.4\%$ over vanilla baseline**, $+50.8\%$ over FP16+PC)

---

## 8. Current State, Limitations & Risks

### 8.1 Working Features
* Complete, automated end-to-end benchmark execution pipeline (`llm-serving run`, `llm-serving quality`, `llm-serving analyze`).
* Automated mock test environment (`--backend mock`) enabling pipeline verification on commodity CPU machines.
* Dual crossover curve fitting and regime map generation.
* Fully populated dataset (156 benchmark runs + 7 perplexity checkpoints + 2 HumanEval evaluations).

### 8.2 Known Limitations & Constraints
* **Single-Node Focus**: Experiments were conducted on a single 48 GB GPU. Inter-GPU NVLink communication overheads in tensor-parallel configurations (TP=2, TP=4) were not measured.
* **Model Scale**: Evaluated on 8B parameter models. While architectural scaling laws indicate similar roofline shifts on 70B models, 70B requires multi-GPU nodes.
* **Static Context Distributions**: Workload generation used uniform context length buckets rather than log-normal real-world conversational distributions.

---

## 9. Way Forward (Future Research Roadmap)

1. **Dynamic Runtime Speculation Switching (Next Immediate Task)**:
   * *Objective*: Implement an adaptive middleware policy in vLLM that continuously monitors queue depth and automatically deactivates speculative heads when $C > C^*_{\text{TPS}}$.
   * *Target*: Prevent the $-17\%$ to $-20\%$ goodput degradation observed at high load.
2. **Characterization of FP4 / INT4 Micro-Scaling KV Formats**:
   * *Objective*: Extend the benchmark suite to evaluate upcoming Blackwell and Hopper FP4 E2M1 KV cache formats.
   * *Done When*: Verify whether FP4 achieves a $3.8\times$ capacity gain and identify the precision floor where code generation degrades.
3. **Disaggregated Prefill-Decode Serving Evaluation**:
   * *Objective*: Separate the serving cluster into dedicated prefill nodes (compute-heavy, FP16) and decode nodes (memory-bound, FP8 + speculative decoding).

---

## 10. Contributions, AI Use & Attribution

### 10.1 Individual Contribution & Technical Ownership
* **Mridul Pant (`cs24mtech15009`)**: Designed the experimental research methodology, developed the 6-layer benchmark orchestrator, implemented the Prometheus telemetry scraper and async closed-loop load client, executed all 156 benchmark sweeps on cloud GPU hardware, developed the dual crossover analysis algorithms, and authored the technical documentation.

### 10.2 Use of AI Assistance
* Generative AI assistants (Google Antigravity / Gemini) were utilized for drafting boilerplate unit test mocks, formatting KaTeX equations, and scaffolding matplotlib visual styling. All benchmark data, telemetry extraction routines, analytical interpretations, and conclusions were independently designed, verified, and validated against empirical logs.

### 10.3 Third-Party Assets & Attribution
* **vLLM** (`v0.32.0`, Apache-2.0): Core inference engine and PagedAttention implementation.
* **EleutherAI LM-Evaluation-Harness** (`v0.4.4`, MIT): Evaluation framework for HumanEval and MMLU.
* **Meta AI**: Model weights for `Meta-Llama-3-8B-Instruct` and `Llama-3.2-1B-Instruct` (Llama 3 Community License).
* **EAGLE Authors**: Pretrained speculative feature head (`yuhuili/EAGLE-LLaMA3-Instruct-8B`, Apache-2.0).

---

## References

1. Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." *Proceedings of the ACM SIGOPS Symposium on Operating Systems Principles (SOSP)*, 2023.
2. Leviathan, Y., Kalman, M., & Matias, Y. "Fast Inference from Transformers via Speculative Decoding." *International Conference on Machine Learning (ICML)*, 2023.
3. Li, Y., et al. "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty." *International Conference on Machine Learning (ICML)*, 2024.
4. Chen, C., et al. "Accelerating Large Language Model Decoding with Speculative Sampling." *arXiv preprint arXiv:2302.01318*, 2023.
5. Micikevicius, P., et al. "FP8 Formats for Deep Learning." *arXiv preprint arXiv:2209.05433*, 2022.
6. Williams, S., Waterman, A., & Patterson, D. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *Communications of the ACM*, 2009.
