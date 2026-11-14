#!/usr/bin/env bash
# Collect the remaining experiments on the GPU instance. Run inside tmux, one step at a time:
#
#   ./scripts/collect_remaining.sh backup          # 1 min   - archive everything first
#   ./scripts/collect_remaining.sh mmlu            # ~30 min - full MMLU, FP16 and FP8 KV
#   ./scripts/collect_remaining.sh prefix_control  # ~20 min - cache-off shared-prefix control
#   ./scripts/collect_remaining.sh finegrid        # ~60 min - C = 8, 32, 128, all methods, both dtypes
#   ./scripts/collect_remaining.sh rerun_fp8       # ~75 min - OPTIONAL clean stage-4 + FP16 EAGLE re-run
#   ./scripts/collect_remaining.sh list            # show what each step will launch (no GPU needed)
#
# Time estimates are rough (scaled from the timings of the budget run). Every step writes
# NEW files (results_extra/, results_budget/lm_eval_mmlu/), so nothing existing is modified.
# The `run` steps are resumable: if one dies, re-run the same command.
set -e

if [ -f "/venv/main/bin/activate" ]; then source /venv/main/bin/activate; fi
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
CLI="python3 -m llm_serving.cli"
CFG=configs/extra.yaml
OUT=results_extra
STEP=${1:?usage: $0 backup|mmlu|prefix_control|finegrid|rerun_fp8|list}

run_stages () {
    mkdir -p "$OUT"
    $CLI run -c "$CFG" --stages "$@" --out "$OUT/results.jsonl" 2>&1 | tee -a "$OUT/run.log"
}

case "$STEP" in
  backup)
    tar czf ~/backup_$(date +%Y%m%d_%H%M).tgz results_budget
    ls -lh ~/backup_*.tgz ;;
  list)
    $CLI list-runs -c "$CFG" ;;
  mmlu)
    # Full MMLU (14,042 questions) for a standard error of ~0.4 points; the 20-per-subject
    # subsample in budget.yaml (~1,140 questions) gives ~1.4 points, too coarse to see an FP8 effect.
    for kv in auto fp8; do
      lm_eval --model vllm \
        --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,dtype=auto,kv_cache_dtype=$kv,gpu_memory_utilization=0.90,max_model_len=4096 \
        --tasks mmlu --batch_size auto \
        --output_path results_budget/lm_eval_mmlu/$kv
    done ;;
  prefix_control)
    run_stages stage5c_prefix_control ;;
  finegrid)
    run_stages stage3b_fine_fp16 stage3b_fine_fp8 ;;
  rerun_fp8)
    run_stages stage4r_spec_fp8 stage3r_eagle_fp16 ;;
  *) echo "unknown step: $STEP"; exit 1 ;;
esac
echo "Finished $STEP at $(date)"
