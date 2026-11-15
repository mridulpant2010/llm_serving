# Architecture Decision Records (ADRs)

**Project Title:** Balancing Memory and Compute in LLM Serving (FP8 KV Cache Precision vs. Speculative Decoding Across Load)  
**Student:** Mridul Pant (`cs24mtech15009`)  
**Course:** CS5903: Distributed AI Training — Final Project  
**Repository:** [github.com/mridulpant2010/llm_serving](https://github.com/mridulpant2010/llm_serving)  

---

## Record Index

* **[ADR-001](#adr-001--methodology-reaffirmation-memory-vs-compute-interaction-under-load)**: Methodology Reaffirmation: Memory vs. Compute Interaction Under Load *(Mandatory 9 October 2026 Record)*
* **[ADR-002](#adr-002--black-box-system-under-test-sut-isolation-with-vllm)**: Black-Box System Under Test (SUT) Isolation with vLLM
* **[ADR-003](#adr-003--speculative-decoding-spectrum-n-gram-eagle-and-independent-draft-model)**: Speculative Decoding Spectrum: N-gram, EAGLE, and Independent Draft Model
* **[ADR-004](#adr-004--dual-metric-crossover-definition-tpot-streaming-vs-tps-goodput)**: Dual-Metric Crossover Definition: TPOT Streaming vs. TPS Goodput
* **[ADR-005](#adr-005--comprehensive-quality-evaluation-protocol-perplexity-and-humaneval)**: Comprehensive Quality Evaluation Protocol: Perplexity and HumanEval

---

### ADR-001 — Methodology Reaffirmation: Memory vs. Compute Interaction Under Load

* **Date:** 2026-10-09  
* **Status:** Accepted *(Reaffirmed)*  
* **Scope:** Core research question, scope boundaries, and milestone roadmap.  
* **Supersedes:** N/A  

#### Context & Decision Drivers
Modern LLM serving systems face two opposing optimizations:
1. **Memory Compression (FP8 KV Cache)**: Halves the per-token KV cache memory footprint, allowing more concurrent requests to fit into VRAM ($C_{\max} \uparrow$) and increasing system throughput.
2. **Compute Acceleration (Speculative Decoding)**: Employs a speculative mechanism to generate multiple candidate tokens per step, verified in parallel by the target model. This trades spare GPU arithmetic capacity for lower per-token latency.

Prior literature benchmarks these levers in isolation or at very low concurrency ($C=1$). However, as concurrency rises, batch sizes grow, saturating GPU Tensor Cores and eliminating the idle arithmetic capacity required for speculative verification. Simultaneously, auxiliary speculative heads consume target GPU memory. On 9 October 2026, the team reviewed whether to shift towards weight quantization (e.g., AWQ/GPTQ) or reaffirm the empirical investigation of memory vs. compute interaction under high load.

#### Options Considered
1. **Option 1 (Shift Scope to Weight Quantization)**: Benchmark 4-bit weights (AWQ, GPTQ) alongside KV cache compression.  
   * *Trade-off*: Squeezes larger models on smaller GPUs, but confounds weight quantization noise with KV cache dynamics and shifts the project into standard model compression rather than distributed serving systems research.
2. **Option 2 (Reaffirm Memory vs. Compute Serving Trade-Off)**: Keep target model weights in native precision (`Meta-Llama-3-8B-Instruct`), isolate the KV cache precision lever (FP16 vs. FP8 E4M3), and systematically measure the speculative decoding crossover point $C^*$ across load ($C \in [1, 256]$) and context length ($512$ to $7,680$).

#### Decision & Rationale
**Reaffirmed Option 2.** The interaction between PagedAttention KV pool sizing and speculative verification overhead under variable concurrency is an open, uncharacterized systems challenge. Retaining unquantized weights isolates KV cache quantization effects and provides clear, uncontaminated crossover boundaries ($C^*$) across load.

#### Consequences & Trade-offs
* **Benefits**: Directly answers the core thesis question; produces a concrete serving regime map dictating when inference engines should dynamically engage or disengage speculation.
* **Costs/Risks**: Requires high-VRAM hardware (48 GB GPU) to evaluate concurrency up to $C=256$ and long contexts ($7,680$ tokens).
* **Impact on Milestones**: Preserves all frozen acceptance tests and milestone deliverables.

#### Evidence & Validation
Preliminary synthetic sweeps confirmed that at $C=1$, speculation yields $1.4\times - 1.5\times$ speedup, but at $C=256$, throughput severely degrades if memory pressure induces preemptions.

#### Revisit Trigger
If production serving engines transition exclusively to disaggregated prefill/decode architectures where decode nodes never batch beyond $C=4$.

---

### ADR-002 — Black-Box System Under Test (SUT) Isolation with vLLM

* **Date:** 2026-10-12  
* **Status:** Accepted  
* **Scope:** Architectural layer design and engine integration.  

#### Context & Decision Drivers
We required a production-grade inference engine supporting PagedAttention, FP8 KV cache kernels, speculative decoding, and automatic prefix caching. We needed to choose between instrumenting engine internals directly or wrapping the engine as a clean, black-box System Under Test.

#### Options Considered
1. **Option 1 (Fork and Instrument vLLM Internals)**: Modify vLLM's internal scheduling loops to emit custom profiling logs.  
   * *Trade-off*: High engineering maintenance, breaks compatibility with upstream releases, and couples benchmark validity to internal modifications.
2. **Option 2 (Black-Box SUT Isolation via OpenAI-Compatible API + Prometheus Metrics)**: Treat vLLM as an immutable subprocess managed via lifecycle wrappers; drive load via asynchronous streaming HTTP clients, and extract GPU telemetry from `/metrics` and NVML.

#### Decision & Rationale
**Selected Option 2.** Black-box isolation ensures that our benchmark results accurately reflect real-world, unmodified vLLM performance (`v0.32.0`). The test harness manages process spawning, socket readiness probes, cold-start warmups, and teardown cleanly.

#### Consequences & Trade-offs
* **Benefits**: Reproducible against any standard vLLM container or upstream version; zero fork maintenance.
* **Costs**: Telemetry must be collected externally via Prometheus metric diffs (`vllm:num_preemptions_total`, `vllm:gpu_cache_usage_factor`).

#### Evidence & Validation
Verified through 24 unit and end-to-end mock server tests (`tests/test_pipeline_mock.py`).

#### Revisit Trigger
If microsecond-level intra-kernel synchronization timestamps become necessary to explain specific CUDA kernel launch stalls.

---

### ADR-003 — Speculative Decoding Spectrum: N-gram, EAGLE, and Independent Draft Model

* **Date:** 2026-10-14  
* **Status:** Accepted  
* **Scope:** Speculation workload selection.  

#### Context & Decision Drivers
Speculative decoding methods differ widely in arithmetic cost, parameter footprint, and acceptance rates. We needed a representative set of speculative paradigms to evaluate how different speculation overheads interact with the KV cache pool.

#### Options Considered
1. **Option 1 (Draft Model Only)**: Test only a smaller autoregressive model (e.g., Llama-3.2-1B).
2. **Option 2 (Three-Tier Hierarchy: N-Gram, EAGLE, and Independent Draft Model)**:
   * *Zero-Parameter*: N-gram prompt lookup (pure matching, 0 extra weights).
   * *Lightweight Head*: EAGLE (`yuhuili/EAGLE-LLaMA3-Instruct-8B`, single transformer layer predicting feature tokens).
   * *Full Autoregressive Draft*: `meta-llama/Llama-3.2-1B-Instruct` (1 billion parameters).

#### Decision & Rationale
**Selected Option 2.** The three-tier hierarchy covers the entire design space: from zero compute/VRAM overhead (N-gram) to moderate overhead (EAGLE) and high overhead (Draft Model). This exposes the "Speculator KV Block Tax," where auxiliary weights shrink the target model's available KV block pool.

#### Consequences & Trade-offs
* **Benefits**: Demonstrates that speculator weight footprint directly degrades the maximum concurrency ceiling $C_{\max}$ (Draft Model consumes $27.4\%$ of target KV cache blocks).
* **Costs**: Requires multi-model loading and checkpoint staging on the serving instance.

#### Evidence & Validation
Empirical measurement confirmed: N-gram VRAM tax: $-0.3\%$, EAGLE VRAM tax: $-10.4\%$, Draft Model VRAM tax: $-27.4\%$.

#### Revisit Trigger
If Medusa or speculative streaming replaces EAGLE as the dominant speculative decoding technique.

---

### ADR-004 — Dual-Metric Crossover Definition: TPOT Streaming vs. TPS Goodput

* **Date:** 2026-10-20  
* **Status:** Accepted  
* **Scope:** Analysis methodology and crossover determination.  

#### Context & Decision Drivers
Literature often ambiguously defines "speedup" in serving. In concurrent serving, latency speedup per stream does not necessarily equal system-level token throughput speedup due to queue admission control.

#### Options Considered
1. **Option 1 (Single Metric: TPOT Speedup Only)**: Define $C^*$ strictly when per-stream decode latency exceeds baseline ($S_{\text{TPOT}}(C^*) = 1.0$).
2. **Option 2 (Dual Metric: $C^*_{\text{TPOT}}$ and $C^*_{\text{TPS}}$)**:
   * **Per-Stream Latency Speedup**: $S_{\text{TPOT}}(C) = \text{TPOT}_{\text{base}}(C) / \text{TPOT}_{\text{spec}}(C)$
   * **System Goodput Speedup**: $S_{\text{TPS}}(C) = \text{TPS}_{\text{spec}}(C) / \text{TPS}_{\text{base}}(C)$

#### Decision & Rationale
**Selected Option 2.** At high concurrency ($C=256$), speculative decoding can shrink active batch concurrency (because each speculative stream requires multiple speculative KV slots), allowing admitted requests to stream quickly ($S_{\text{TPOT}} = 2.15\times$) while overall system throughput suffers ($S_{\text{TPS}} = 0.80\times$) due to waiting queue delays. Reporting both metrics resolves this paradox and provides actionable guidelines for both latency-critical and throughput-critical SLAs.

#### Consequences & Trade-offs
* **Benefits**: Prevents misleading claims about speculative benefits; accurately models admission controller dynamics.
* **Costs**: Requires dual crossover detection algorithms and dual plotting pipelines in `src/llm_serving/analysis.py`.

#### Evidence & Validation
Empirical data at $C=256$: FP8 Draft Model delivers $S_{\text{TPOT}} = 2.15\times$, but $S_{\text{TPS}} = 0.80\times$, matching the theoretical model.

#### Revisit Trigger
If serving schedulers eliminate speculative slots during decoding dynamically within the same forward pass.

---

### ADR-005 — Comprehensive Quality Evaluation Protocol: Perplexity and HumanEval

* **Date:** 2026-10-25  
* **Status:** Accepted  
* **Scope:** Quality assessment protocol and kernel validation.  

#### Context & Decision Drivers
Quantizing the KV cache to FP8 E4M3 halves memory consumption, but critics frequently question numerical stability and loss of reasoning fidelity over long contexts. Standard HuggingFace evaluation scripts compute perplexity using standard FP16 PyTorch attention, failing to exercise vLLM's real FP8 PagedAttention CUDA kernels.

#### Options Considered
1. **Option 1 (Offline HuggingFace Perplexity Only)**: Quick, but invalid because it bypasses vLLM's real runtime FP8 quantization kernels.
2. **Option 2 (In-Engine Perplexity via `prompt_logprobs` + Downstream Execution via HumanEval)**:
   * Evaluate token-weighted negative log-likelihood (NLL) directly through vLLM's server using `prompt_logprobs` across context windows up to 7,680 tokens.
   * Evaluate full functional correctness on all 164 coding tasks of the **HumanEval** benchmark (`pass@1`).

#### Decision & Rationale
**Selected Option 2.** Evaluating directly through vLLM exercises the exact FP8 dynamic scaling and quantization kernels used in serving. Pairing perplexity (language modeling likelihood) with HumanEval (multi-step algorithmic code execution) provides an airtight, two-tier proof of quality preservation.

#### Consequences & Trade-offs
* **Benefits**: Proven empirical Pareto dominance: $\le +0.26\%$ WikiText-2 perplexity delta across 491,456 tokens, and HumanEval `pass@1` of $58.54\%$ (FP8) vs. $59.76\%$ (FP16), within standard error ($\pm 3.85\%$).
* **Costs**: Required patching `lm-eval` stop sequence limits to conform to OpenAI API schemas (`max_length=4`).

#### Evidence & Validation
Full 164-task evaluation logs verified and stored in `results_budget/quality.jsonl`.

#### Revisit Trigger
If 4-bit KV cache (FP4/INT4) becomes supported in upstream vLLM kernels.
