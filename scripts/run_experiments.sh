#!/usr/bin/env bash
# Run the benchmark stages. Run inside tmux. Results are append-only JSONL and
# resumable: if the job dies, re-run the same command and finished rows are skipped.
#
#   ./scripts/run_experiments.sh pilot    # ~45-60 min, checks every config launches
#   ./scripts/run_experiments.sh budget   # ~4 h, reduced matrix that fits a $5 budget
#   ./scripts/run_experiments.sh full     # full matrix, 3 reps (~12-15 h on one 48 GB GPU)
#   ./scripts/run_experiments.sh analyze <results_dir>
set -e
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
CLI="python -m llm_serving.cli"
MODE=${1:-pilot}

run_suite () {
    local cfg=$1 out=$2 lmeval_flag=$3
    mkdir -p "$out"
    echo "=== [$MODE] config=$cfg  -> $out ==="
    $CLI list-runs -c "$cfg"
    $CLI run     -c "$cfg" --out "$out/results.jsonl"   2>&1 | tee -a "$out/run.log"
    $CLI quality -c "$cfg" --out "$out/quality.jsonl" $lmeval_flag 2>&1 | tee -a "$out/run.log"
    $CLI analyze --results "$out/results.jsonl" --quality "$out/quality.jsonl" --out "$out/report"
    echo "Report: $out/report/summary.md"
}

case "$MODE" in
    pilot)  run_suite configs/pilot.yaml        results_pilot  --skip-lm-eval ;;
    budget) run_suite configs/budget.yaml       results_budget "" ;;
    full)   run_suite configs/experiments.yaml  results_full   "" ;;
    analyze)
        d=${2:?usage: analyze <results_dir>}
        $CLI analyze --results "$d/results.jsonl" --quality "$d/quality.jsonl" --out "$d/report" ;;
    *) echo "Usage: $0 [pilot|budget|full|analyze <dir>]"; exit 1 ;;
esac
echo "Finished at $(date)"
