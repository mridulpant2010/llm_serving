# Open issues (copy each into a GitHub issue)

Each item below is an open task from the project report, section 9. Create them with the GitHub CLI or the web form; labels suggested in brackets.

## 1. Run full MMLU for FP16 and FP8 KV cache [evaluation]
Starting point: `scripts/collect_remaining.sh mmlu` (about 30-40 minutes of GPU time).
Done when: two result files exist and the accuracy difference is reported with a confidence interval. Today only perplexity and HumanEval cover quality, so knowledge tasks are untested.

## 2. Finer concurrency grid at C = 8, 32, 128 [evaluation]
Starting point: stages `stage3b_fine_fp16` and `stage3b_fine_fp8` in `configs/extra.yaml`.
Done when: the concurrency where speculative speedup crosses 1.0 is located within one grid step, per method and dtype. Today only C = 1, 4, 16, 64, 256 are measured, so no crossover is quoted.

## 3. Find out why the FP8 draft-model run is slow at low concurrency [investigation]
Observed: the draft model gives 1.50x at C=1 with FP16 KV cache but 1.03x with FP8, at equal acceptance (52%).
Done when: the slow step is identified in a profiler trace, or the effect is shown to disappear on another vLLM version.

## 4. Find the source of the placeholder telemetry in the first run [bug]
Observed: 64 rows of the first run carried constant TTFT (500 ms) and power (298 W) after a truncated download and rebuild. Look in `src/llm_serving/orchestrator.py` and `telemetry.py`.
Done when: a new test fails if TTFT or power is constant across a sweep.

## 5. Revoke the committed SSH key at the GPU provider [security]
The private key was committed early on and later untracked, but it is still in git history.
Done when: the key is revoked at the provider and a new one is not stored in the repository.

```
gh issue create --title "Run full MMLU for FP16 and FP8 KV cache" --body-file docs/open_issues.md
```
(Create one issue per section; the command above is an example only.)
