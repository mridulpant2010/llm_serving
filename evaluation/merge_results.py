"""Merge the original budget run and the clean re-run into one analysis dataset.

usage: python evaluation/merge_results.py [orig_dir] [rerun_dir] [out_dir]
       defaults: results_budget results_extra results_final

Rules (see ADR-007):
  * Rows rebuilt from run.log (placeholder signature: TTFT p50 == 500 and power == 298 W) are never
    used for analysis. They are compared against the clean re-run in rebuilt_vs_rerun.csv.
  * Stage 1 (FP16 vs FP8 under load) and the FP16 N-gram sweep (plus its FP16 no-speculation
    baseline, used only to pair the N-gram speedups) come from the original run.
  * Stages 3, 4 and 5 come from the clean re-run; stage names are normalised so the analysis
    scripts see one naming scheme.
  * Failed launches (status != ok) are dropped.
Every output row carries a `source` field: original or rerun.
"""
import sys, json, os, shutil, collections, statistics as st
O = (sys.argv[1] if len(sys.argv) > 1 else "results_budget").rstrip("/") + "/"
R = (sys.argv[2] if len(sys.argv) > 2 else "results_extra").rstrip("/") + "/"
OUT = (sys.argv[3] if len(sys.argv) > 3 else "results_final").rstrip("/") + "/"
os.makedirs(OUT + "server_logs", exist_ok=True)
load = lambda p: [json.loads(l) for l in open(p) if l.strip()]
placeholder = lambda r: r.get("ttft_ms_p50") == 500 and r.get("gpu_power_w_mean") == 298
NORM = {"stage3r_spec_fp16": "stage3_spec_fp16", "stage4r_spec_fp8": "stage4_spec_fp8",
        "stage5r_prefix_on": "stage5_prefix_on", "stage5c_prefix_control": "stage5_cache_off"}

orig_all = [r for r in load(O + "results.jsonl") if r.get("status") == "ok"]
orig_real = [r for r in orig_all if not placeholder(r)]
final = []
for r in orig_real:
    if r["stage"] == "stage1_kv_capacity" or (r["stage"] == "stage3_spec_fp16" and r["spec"] in ("none", "ngram")):
        final.append(dict(r, source="original"))
rerun = [r for r in load(R + "results.jsonl") if r.get("status") == "ok"]
assert not any(placeholder(r) for r in rerun), "placeholder rows in re-run"
for r in rerun:
    final.append(dict(r, stage=NORM[r["stage"]], source="rerun"))
with open(OUT + "results.jsonl", "w") as f:
    for r in final: f.write(json.dumps(r) + "\n")
shutil.copy(O + "quality.jsonl", OUT + "quality.jsonl")
for fn in os.listdir(R + "server_logs"):
    for a, b in NORM.items():
        if fn.startswith(a):
            shutil.copy(R + "server_logs/" + fn, OUT + "server_logs/" + fn.replace(a, b, 1))
shutil.copy(O + "server_logs/stage3_spec_fp16__kv-auto__spec-ngram__pc-0.log", OUT + "server_logs/")

# how well did the rebuilt rows agree with the clean re-run? (throughput, the field that was measured)
def mean_tps(rows, key):
    d = collections.defaultdict(list)
    for r in rows: d[key(r)].append(r["throughput_tps"])
    return {k: st.mean(v) for k, v in d.items()}
k_o = lambda r: (NORM.get(r["stage"], r["stage"]), r["kv_dtype"], r["spec"], r["concurrency"], r["context_len"])
rb = [r for r in orig_all if placeholder(r)]
A = mean_tps(rb, k_o); B = mean_tps([dict(r, stage=NORM[r["stage"]]) for r in rerun], k_o)
# the original prefix stage has no separate cache-on name; map it to the re-run's cache-on stage
A = {(("stage5_prefix_on" if k[0] == "stage5_prefix_ablation" else k[0]),) + k[1:]: v for k, v in A.items()}
lines = ["stage,kv_dtype,spec,concurrency,context_len,rebuilt_tok_s,rerun_tok_s,rerun_vs_rebuilt_pct"]
diffs = []
for k in sorted(A, key=str):
    if k in B:
        d = (B[k] / A[k] - 1) * 100; diffs.append(abs(d))
        lines.append(",".join(map(str, k)) + f",{A[k]:.1f},{B[k]:.1f},{d:+.1f}")
open(OUT + "rebuilt_vs_rerun.csv", "w").write("\n".join(lines) + "\n")
print(f"rows: original {sum(r['source']=='original' for r in final)}, rerun {sum(r['source']=='rerun' for r in final)}, total {len(final)}")
print(f"rebuilt vs rerun: {len(diffs)} matching cells, median |diff| {st.median(diffs):.1f}%, max {max(diffs):.1f}%, within 5%: {sum(d<=5 for d in diffs)}")
