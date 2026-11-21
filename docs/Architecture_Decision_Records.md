# Architecture Decision Records

**Project:** Balancing Memory and Compute in LLM Serving (FP8 KV cache and speculative decoding), Project ID 24, solo (Mridul Pant, cs24mtech15009).
**Repository:** https://github.com/mridulpant2010/llm_serving
**Log written:** 2026-11-15, finalised 2026-11-21. For ADR-001 to ADR-005 the dates below are the *earliest evidence in the git history* (commit and date), not remembered decision dates, and nothing is backdated. ADR-006 to ADR-008 were decided on the dates shown.

| ID | Title | Status |
|---|---|---|
| ADR-001 | Methodology: reaffirmed by 9 October 2026 | Accepted |
| ADR-002 | Treat vLLM as a black-box system under test | Accepted |
| ADR-003 | Three speculators with the same k=4 | Accepted |
| ADR-004 | Dual-metric crossover (TPOT and throughput) | **Superseded by ADR-006** |
| ADR-005 | Quality protocol: in-engine perplexity and HumanEval | Accepted, with limits |
| ADR-006 | Throughput speedup is the primary metric; no headline C* | Accepted |
| ADR-007 | Use only measured fields from the 64 rebuilt rows | Accepted |
| ADR-008 | Reduced budget matrix, and early stop when credit ran out | Accepted |

---

## ADR-001: Methodology, reaffirmed by 9 October 2026

- **Date, status, scope:** Plan first recorded 2026-10-04 (commit `5428592`); this record written 2026-11-21. Accepted. Whole project: research question, scope, milestones. Supersedes: none.
- **Context and drivers:** Two serving optimizations interact. FP8 KV cache raises capacity; speculation spends spare compute. Prior work studies them separately, and the project needs a result that holds under load on one GPU. No review of this decision with the mentor is recorded in the repository.
- **Options:** (1) Keep the approach: measure FP8 and three speculators on a real server across load. (2) Shift to weight quantization (AWQ or GPTQ) next to KV compression. (3) Build an analytic model only.
- **Decision and rationale:** **Reaffirmed (not changed).** The research question, model and two levers recorded in the plan documents on 2026-10-04 are the ones submitted, so the 9 October (11:59 PM IST) status is: unchanged. Option 1 as written in the plan: it isolates the KV lever (weights stay unquantized) and uses a real scheduler, which a model cannot capture. This is judgment, not measurement.
- **Consequences:** Needs a GPU with enough memory for 256 concurrent requests, which drove the budget matrix (ADR-008). Weight quantization and multi-GPU are out of scope. No change to the approved scope is recorded between 2026-10-04 and submission.
- **Evidence:** Plan documents (`subtopic_selection.md`, `project_analysis.md`, `system_design.md`, `project_report.md`) and the first harness code committed together on 2026-10-04 (`5428592`), five days before the 9 October cutoff; sweeps and results in `results_budget/`. No contemporaneous written reaffirmation to the mentor is on record, and this record was written now, not at the time.
- **Revisit trigger:** A result that FP8 KV alone gives under 10% throughput change at every tested load, which would make the comparison uninteresting.

## ADR-002: Treat vLLM as a black-box system under test

- **Date, status, scope:** Earliest evidence 2026-10-04 (`5428592`: `server.py`, `mock_server.py`). Accepted. Harness architecture.
- **Context:** The study needs FP8 KV cache, three speculators and prefix caching from one production engine.
- **Options:** (1) Fork vLLM and add instrumentation. (2) Run vLLM unmodified, drive it through its API and read `/metrics`, server logs and NVML.
- **Decision and rationale:** Option 2. Results then describe unmodified vLLM 0.32.0, and nothing needs rebasing on upstream.
- **Consequences:** Telemetry is limited to what vLLM exposes (preemption counters, queue depth, acceptance counters). Per-kernel timing is not available.
- **Evidence:** `src/llm_serving/server.py`, `telemetry.py`; 24 automated tests (`tests/`) run against a mock server; real runs in `results_budget/server_logs/`.
- **Revisit trigger:** A question that needs kernel-level timing to answer.

## ADR-003: Three speculators with the same k=4

- **Date, status, scope:** Earliest evidence 2026-10-04 (`5428592`); the k=4 settings are in `configs/budget.yaml` from 2026-11-02 (`39f88d8`). Accepted. Speculation matrix.
- **Context:** Speculators differ in extra weights and acceptance. A fair comparison needs a spread of cost.
- **Options:** (1) Draft model only. (2) Draft model (Llama-3.2-1B), EAGLE head, N-gram lookup, each proposing k=4 tokens.
- **Decision and rationale:** Option 2. It spans no extra weights (N-gram), a small head (EAGLE) and a full second model.
- **Consequences:** The extra weights reduce the KV pool: -27% (draft), -10% (EAGLE), -0.3% (N-gram), measured from server logs. A single k hides each method's own best k; not tuned.
- **Evidence:** `configs/budget.yaml`; block counts in `results_budget/run.log`; acceptance counters in the server logs.
- **Revisit trigger:** Evidence that another k changes which method wins at C=16.

## ADR-004: Dual-metric crossover (TPOT and throughput)

- **Date, status, scope:** Earliest evidence 2026-10-04 (`analysis.py`, `5428592`); dual-crossover analysis committed 2026-11-09 (`b50b838`). **Superseded by ADR-006** on 2026-11-14. Analysis method. Kept for traceability.
- **Context:** In concurrent serving, a per-token latency speedup can differ from a throughput speedup.
- **Options:** (1) TPOT speedup only. (2) Report both a TPOT-based and a throughput-based crossover concurrency C*.
- **Decision and rationale:** Option 2, interpolating each C* between measured levels.
- **Consequences and why superseded:** With 5 levels and 2 repeats, the interpolated C* values moved with the interpolation rule, and TPOT disagreed with throughput at C=256 because of the KV-pool tax. The README and an earlier report quoted figures that did not agree with each other.
- **Evidence:** `evaluation/make_analysis.py` (paired speedups).
- **Revisit trigger:** A denser grid (ADR-008 follow-up).

## ADR-005: Quality protocol, in-engine perplexity and HumanEval

- **Date, status, scope:** Earliest evidence 2026-10-04 (`quality_eval.py`, `5428592`); perplexity fixes 2026-11-02; HumanEval direct vLLM mode 2026-11-14 (`e235675`). Accepted, with limits. Quality evaluation.
- **Context:** An offline perplexity script would not exercise vLLM's FP8 attention kernels.
- **Options:** (1) Offline Hugging Face perplexity. (2) Perplexity through the running vLLM server (`prompt_logprobs`) on 64 WikiText-2 windows per context length, plus HumanEval (164 problems) through lm-eval in direct vLLM mode (`scripts/run_humaneval.py`, `max_model_len` 4096). (3) Add MMLU.
- **Decision and rationale:** Option 2 now; MMLU planned but not run.
- **Consequences:** FP8 costs at most 0.26% perplexity. HumanEval is 98 versus 96 of 164, which cannot show equivalence at n=164. FP16 at 7,680 crashed, so that tier has no pair. General-knowledge quality (MMLU) is untested.
- **Evidence:** `results_budget/quality.jsonl`; `results_budget/lm_eval/`.
- **Revisit trigger:** MMLU results, or a context length above 4,096 where FP16 can be run.

## ADR-006: Throughput speedup is the primary metric; no headline C*

- **Date, status, scope:** 2026-11-14. Accepted. Supersedes ADR-004. Analysis and all reported claims.
- **Context and evidence for change:** Re-analysis of `results.jsonl` showed TPOT-based crossovers disagree with throughput at C=256 for the draft model and EAGLE, because a smaller KV pool admits fewer requests. Old approach (TPOT-based C*) to new evidence (pool sizes from the logs, 2-repeat ranges) to the selected approach below.
- **Options:** (1) Keep TPOT-based C*. (2) Throughput speedup per repeat, reported with its range, as a regime map by concurrency and method. (3) Fit a smooth curve and report C*.
- **Decision and rationale:** Option 2. A regime map states only what was measured. This is a presentation decision backed by the measurements.
- **Consequences:** The headline becomes "speculation pays at C of 4 or below (draft model to about 16)"; no interpolated crossover is quoted. Slides, README and report were aligned. Cost: no single number per method.
- **Impact on milestones and tests:** The crossover milestone is met as a measured regime map instead of an interpolated C*. (The frozen acceptance wording is not recorded in the repository.)
- **Revisit trigger:** The finer C grid (8, 32, 128) and 3 repeats are run.

## ADR-007: Use only measured fields from the 64 rebuilt rows

- **Date, status, scope:** Rebuild committed on or before 2026-11-09 (commit `b50b838`); analysis rule decided 2026-11-14. Accepted. Data handling for rows 93-156 of `results.jsonl` (Stage 3 draft model at C=256 and EAGLE, all of Stages 4 and 5).
- **Context:** The `results.jsonl` downloaded from the first Vast.ai instance was truncated at 92 lines, while `run.log` and `summary.json` held one summary line per run for all stages. The 64 missing rows were rebuilt from those lines. Throughput, TPOT p50, KV blocks, preemptions and mean acceptance are measured values; TTFT, power, GPU utilisation and memory, KV usage, running and waiting counts and timestamps are placeholders or estimates.
- **Options:** (1) Use all fields as recorded. (2) Use only the measured fields, corroborate acceptance with the server logs, and disclose the rebuild. (3) Discard Stages 3 (EAGLE), 4 and 5. (4) Re-run them (not possible: no GPU credit).
- **Decision and rationale:** Option 2. Placeholders carry no information; discarding the stages would remove the FP8 speculation sweep, which is the core of the study.
- **Consequences:** No TTFT, energy or utilisation claims from these rows; throughput and TPOT have only 0.1 precision; the raw rows cannot be recovered; this is disclosed in the report (§8) and the slides.
- **Evidence:** `results_budget/run.log`; rows 93-156 of `results_budget/results.jsonl` (placeholder signature: TTFT p50 500, power 298 W, timestamps 1793600000 to 1793600063); `evaluation/make_analysis.py`.
- **Revisit trigger:** The clean re-run (`stage4r_spec_fp8`, `stage3r_eagle_fp16` in `configs/extra.yaml`) is executed.

## ADR-008: Reduced budget matrix, and early stop when credit ran out

- **Date, status, scope:** Budget matrix first committed 2026-11-02 (`39f88d8`); early stop 2026-11-14. Accepted. Experiment coverage.
- **Context:** About 4 GPU-hours were available. The full matrix needs 12 to 15.
- **Options:** (1) Full matrix. (2) Budget matrix: 2 repeats, 5 concurrency levels, 3 contexts in Stage 1 (`configs/budget.yaml`). (3) Also run the extra stages after the budget matrix.
- **Decision and rationale:** Option 2 first. When GPU credit ran out on 2026-11-14, Option 3 was **not** done; the extra stages are specified and ready (`configs/extra.yaml`, `scripts/collect_remaining.sh`).
- **Consequences:** MMLU, the cache-off prefix control, the finer grid and a clean Stage 4 re-run are missing and are listed as not run in the report. All claims are limited to the 5-level, 2-repeat data.
- **Evidence:** `configs/budget.yaml` header; `configs/extra.yaml`.
- **Revisit trigger:** Any new GPU credit; start with MMLU and the cache-off control.
