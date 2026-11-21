# evaluation/

Regenerates every figure and table used in `docs/Project_Report.md` from the raw benchmark output.

```bash
pip install pandas numpy matplotlib openpyxl
python evaluation/make_analysis.py results_budget evaluation/output     # charts 01-10, comparison_tables.xlsx
python evaluation/make_regime_map.py results_budget evaluation/output   # chart 11 (regime map)
```

Inputs: `results_budget/results.jsonl`, `results_budget/quality.jsonl`, `results_budget/run.log`, `results_budget/server_logs/`.
Outputs: PNG charts and `comparison_tables.xlsx` (sheets: Stage1_FP16_vs_FP8, Speculation_speedups, Best_config_by_C, Speculator_tax).

Method notes: speedup is computed per repeat (speculative divided by the same-dtype baseline at the same C) and reported with the min/max of the two repeats. Acceptance and verification waste come from the `SpecDecoding metrics` lines in the server logs. KV block counts come from `run.log`. Rows 93-156 of `results.jsonl` (Stage 3 draft model C=256 and EAGLE, Stages 4 and 5) were rebuilt from `run.log`; their telemetry fields other than throughput, TPOT p50, KV blocks, preemptions and acceptance are placeholders and are not used (ADR-007).

Follow-up runs (not yet executed): `configs/extra.yaml`, `scripts/collect_remaining.sh`.
