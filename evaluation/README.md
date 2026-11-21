# evaluation/

Regenerates every figure and table used in `docs/Project_Report.md` from the raw benchmark output.

```bash
pip install pandas numpy matplotlib openpyxl
python evaluation/merge_results.py results_budget results_extra results_final   # merged dataset + rebuilt_vs_rerun.csv
python evaluation/make_analysis.py results_final evaluation/output     # charts 01-10, comparison_tables.xlsx
python evaluation/make_regime_map.py results_final evaluation/output   # chart 11 (regime map)
```

Inputs: `results_budget/` (first run) and `results_extra/` (clean re-run): `results.jsonl`, `quality.jsonl`, `server_logs/`.
Outputs: PNG charts and `comparison_tables.xlsx` (sheets: Prefix_cache_on_vs_off, Rebuilt_vs_rerun, Stage1_FP16_vs_FP8, Speculation_speedups, Best_config_by_C, Speculator_tax).

Method notes: `merge_results.py` keeps Stage 1 and the FP16 N-gram sweep (with its own FP16 baseline, used only for pairing) from the first run, and Stages 3, 4 and 5 from the re-run; it drops failed launches and the 64 rows that had been rebuilt from `run.log` (placeholder signature: TTFT p50 500 and power 298 W), and writes `rebuilt_vs_rerun.csv` comparing them with the clean data (ADR-007, ADR-009). Speedup is computed per repeat (speculative divided by the same-dtype baseline at the same C and from the same source) and reported with the min/max of the repeats. Acceptance and verification waste come from the `SpecDecoding metrics` lines in the server logs. KV block counts come from the `kv_blocks` field of the rows.

Not yet executed: MMLU (`scripts/collect_remaining.sh mmlu`) and the finer concurrency grid (`stage3b_*` in `configs/extra.yaml`).
