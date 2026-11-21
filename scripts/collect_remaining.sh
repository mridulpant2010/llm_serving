#!/usr/bin/env bash
# Collect the remaining experiments on the GPU instance. Run inside tmux, ONE step at a time:
#
#   ./scripts/collect_remaining.sh env             # 1 min   - record GPU, driver, CUDA, vLLM, torch, lm-eval versions
#   ./scripts/collect_remaining.sh list            # dry run: what each stage will launch (no GPU work)
#   ./scripts/collect_remaining.sh rerun           # ~1h45  - re-run only the rebuilt rows: stage 4 (FP8) + stage 3 EAGLE/draft/baseline (FP16)
#   ./scripts/collect_remaining.sh mmlu            # ~40 min - full MMLU, FP16 and FP8 KV
#   ./scripts/collect_remaining.sh prefix          # ~40 min - OPTIONAL: stage 5 re-run (cache on) + cache-off control
#   ./scripts/collect_remaining.sh finegrid        # ~60 min - OPTIONAL: C = 8, 32, 128
#   ./scripts/collect_remaining.sh verify          # check every results file is complete and valid JSON
#   ./scripts/collect_remaining.sh pack            # tar + sha256 for download (run LAST, after verify)
#
# Everything goes to NEW files (results_extra/, results_budget/lm_eval_mmlu/); nothing existing
# is modified. The `run` steps are resumable: if one dies, re-run the same command.
set -eo pipefail

if [ -f "/venv/main/bin/activate" ]; then source /venv/main/bin/activate; fi
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
CLI="python3 -m llm_serving.cli"
CFG=configs/extra.yaml
OUT=results_extra
STEP=${1:?usage: $0 env|list|rerun|mmlu|prefix|finegrid|verify|pack}
mkdir -p "$OUT"

run_stages () {
    $CLI run -c "$CFG" --stages "$@" --out "$OUT/results.jsonl" 2>&1 | tee -a "$OUT/run.log"
}

case "$STEP" in
  env)
    {
      echo "date: $(date -u)"; hostname
      nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
      nvidia-smi | grep -i "cuda version" || true
      python3 -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0), '| sm', torch.cuda.get_device_capability(0))"
      pip show lm_eval 2>/dev/null | grep -i -E "^(name|version)" || pip show lm-eval | grep -i -E "^(name|version)"
      python3 --version
      git rev-parse HEAD
    } 2>&1 | tee "$OUT/env.txt"
    pip freeze > "$OUT/pip_freeze.txt" ;;
  list)
    $CLI list-runs -c "$CFG" ;;
  rerun)
    run_stages stage4r_spec_fp8
    run_stages stage3r_spec_fp16 ;;
  mmlu)
    for kv in auto fp8; do
      lm_eval --model vllm \
        --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,dtype=auto,kv_cache_dtype=$kv,gpu_memory_utilization=0.90,max_model_len=4096 \
        --tasks mmlu --batch_size auto \
        --output_path results_budget/lm_eval_mmlu/$kv 2>&1 | tee -a "$OUT/mmlu_$kv.log"
    done ;;
  prefix)
    run_stages stage5r_prefix_on stage5c_prefix_control ;;
  finegrid)
    run_stages stage3b_fine_fp16 stage3b_fine_fp8 ;;
  verify)
    python3 - "$OUT/results.jsonl" <<'PY'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]          # fails loudly on a broken line
ok = [r for r in rows if r.get("status") == "ok"]
print(f"{len(rows)} rows, {len(ok)} ok, {len(rows)-len(ok)} failed")
for k, v in sorted(collections.Counter((r["stage"], r["kv_dtype"], r["spec"]) for r in rows).items()):
    print(f"  {k}: {v}")
const = [r for r in ok if r.get("ttft_ms_p50") == 500.0 and r.get("gpu_power_w_mean") == 298.0]
print("placeholder-looking rows:", len(const), "(must be 0)")
PY
    find results_budget/lm_eval_mmlu -name 'results_*.json' 2>/dev/null || echo "no MMLU results yet" ;;
  pack)
    tar czf ~/results_extra.tgz results_extra results_budget/lm_eval_mmlu 2>/dev/null || tar czf ~/results_extra.tgz results_extra
    sha256sum ~/results_extra.tgz | tee ~/results_extra.tgz.sha256
    ls -lh ~/results_extra.tgz ;;
  *) echo "unknown step: $STEP"; exit 1 ;;
esac
echo "Finished $STEP at $(date)"
