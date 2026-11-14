"""
Generate publication-quality charts from the complete Vast.ai RTX PRO 5000 Blackwell benchmark data.
All data points match the exact median measurements in results_budget/results.jsonl and quality.jsonl.
"""

import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

out_dir = Path("c:/coding/llm_serving/results_budget/report")
out_dir.mkdir(parents=True, exist_ok=True)
artifact_dir = Path(r"C:\Users\pantm\.gemini\antigravity\brain\87a0ce35-a3e8-4c43-8b04-b9d93ae5d821")

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

concurrency = [1, 4, 16, 64, 256]

# -------------------------------------------------------------
# Figure 1: Throughput vs Concurrency (FP16 vs FP8 Knee Shift)
# -------------------------------------------------------------
tps_fp16_512  = [74.2, 265.4, 894.8, 2189.6, 2835.6]
tps_fp8_512   = [75.0, 269.2, 925.9, 2400.5, 3411.5]

tps_fp16_2048 = [70.5, 227.7, 567.1, 857.2, 782.0]
tps_fp8_2048  = [71.7, 235.5, 607.1, 1021.9, 1136.5]

tps_fp16_7680 = [59.3, 142.4, 220.4, 221.2, 219.1]
tps_fp8_7680  = [61.4, 152.9, 249.6, 296.1, 302.1]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=False)

# Subplot 1: Context 512
axes[0].plot(concurrency, tps_fp16_512, "o--", color="#1f77b4", linewidth=2.5, markersize=8, label="FP16 KV (auto)")
axes[0].plot(concurrency, tps_fp8_512, "s-", color="#2ca02c", linewidth=2.5, markersize=8, label="FP8 KV")
axes[0].set_xscale("log", base=2)
axes[0].set_title("Context = 512 tokens (Compute Bound)")
axes[0].set_xlabel("Concurrency (C)")
axes[0].set_ylabel("Throughput (tokens / sec)")
axes[0].legend(loc="upper left")
axes[0].annotate("+20.3% Peak TPS\n(3,411 vs 2,836)", xy=(256, 3411.5), xytext=(20, 2950),
                 arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6))

# Subplot 2: Context 2048
axes[1].plot(concurrency, tps_fp16_2048, "o--", color="#1f77b4", linewidth=2.5, markersize=8, label="FP16 KV (auto)")
axes[1].plot(concurrency, tps_fp8_2048, "s-", color="#2ca02c", linewidth=2.5, markersize=8, label="FP8 KV")
axes[1].set_xscale("log", base=2)
axes[1].set_title("Context = 2048 tokens (Knee Shift & Thrashing)")
axes[1].set_xlabel("Concurrency (C)")
axes[1].legend(loc="upper left")
axes[1].annotate("FP16 Thrashing Drop\n(-8.8%, 35 preemptions)", xy=(256, 782.0), xytext=(22, 520),
                 arrowprops=dict(facecolor="red", shrink=0.08, width=1, headwidth=6))
axes[1].annotate("FP8 Sustained\n(+45.3% at C=256, 0 preempts)", xy=(256, 1136.5), xytext=(16, 1040),
                 arrowprops=dict(facecolor="green", shrink=0.08, width=1, headwidth=6))

# Subplot 3: Context 7680
axes[2].plot(concurrency, tps_fp16_7680, "o--", color="#1f77b4", linewidth=2.5, markersize=8, label="FP16 KV (auto)")
axes[2].plot(concurrency, tps_fp8_7680, "s-", color="#2ca02c", linewidth=2.5, markersize=8, label="FP8 KV")
axes[2].set_xscale("log", base=2)
axes[2].set_title("Context = 7680 tokens (Memory Wall)")
axes[2].set_xlabel("Concurrency (C)")
axes[2].legend(loc="upper left")
axes[2].annotate("Knee C=16 (FP16)\n220.4 tok/s", xy=(16, 220.4), xytext=(3, 165),
                 arrowprops=dict(facecolor="blue", shrink=0.08, width=1, headwidth=6))
axes[2].annotate("Knee C=64 (FP8)\n+36.6% Ceiling (302.1 tok/s)", xy=(64, 296.1), xytext=(8, 285),
                 arrowprops=dict(facecolor="green", shrink=0.08, width=1, headwidth=6))

plt.tight_layout()
f1 = out_dir / "figure1_kv_throughput_knee.png"
plt.savefig(f1)
plt.close()

# -------------------------------------------------------------
# Figure 2: Speculative Decoding Crossover C* (Dual View: TPOT vs System TPS)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), sharey=False)

# Panel A: Active Per-Token Speedup S_TPOT(C) = TPOT_base / TPOT_spec (FP16 vs FP8)
s_tpot_draft_fp16 = [1.511, 1.417, 1.232, 1.036, 1.444]
s_tpot_eagle_fp16 = [1.015, 1.039, 1.004, 0.896, 0.974]
s_tpot_ngram_fp16 = [1.092, 1.117, 1.043, 0.921, 0.829]

s_tpot_draft_fp8  = [1.343, 1.357, 1.237, 1.038, 2.154]
s_tpot_eagle_fp8  = [1.077, 1.095, 1.080, 0.969, 1.317]
s_tpot_ngram_fp8  = [1.220, 1.163, 1.170, 1.001, 1.234]

axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Parity S(C) = 1.0")
axes[0].plot(concurrency, s_tpot_draft_fp16, "o--", color="#d62728", lw=2, ms=7, label=r"Draft 1B (FP16, $\alpha=52.5\%$)")
axes[0].plot(concurrency, s_tpot_draft_fp8,  "o-",  color="#d62728", lw=2.5, ms=8, label=r"Draft 1B (FP8, $\alpha=52.8\%$)")
axes[0].plot(concurrency, s_tpot_eagle_fp16, "^--", color="#ff7f0e", lw=2, ms=7, label=r"EAGLE (FP16, $C^*=16.9$)")
axes[0].plot(concurrency, s_tpot_eagle_fp8,  "^-",  color="#ff7f0e", lw=2.5, ms=8, label=r"EAGLE (FP8, $C^*=43.9$)")
axes[0].plot(concurrency, s_tpot_ngram_fp16, "s--", color="#9467bd", lw=2, ms=7, label=r"N-gram (FP16, $C^*=25.9$)")
axes[0].plot(concurrency, s_tpot_ngram_fp8,  "s-",  color="#9467bd", lw=2.5, ms=8, label=r"N-gram (FP8, No Cross)")

axes[0].axvline(16.9, color="#ff7f0e", linestyle=":", alpha=0.7)
axes[0].axvline(43.9, color="#ff7f0e", linestyle="-.", alpha=0.9)
axes[0].annotate(r"EAGLE $C^*$ Shift" + "\n" + r"$16.9 \rightarrow 43.9$ (+2.6x)",
                 xy=(43.9, 1.0), xytext=(32, 1.55),
                 arrowprops=dict(facecolor="#ff7f0e", shrink=0.06, width=1.5, headwidth=6),
                 fontweight="bold")
axes[0].set_xscale("log", base=2)
axes[0].set_xlabel("Concurrency (C)")
axes[0].set_ylabel(r"Stream Speedup $S_{\mathrm{TPOT}}(C) = \mathrm{TPOT}_{\mathrm{base}} / \mathrm{TPOT}_{\mathrm{spec}}$")
axes[0].set_title("A. Per-Stream Decode Speedup (TPOT Ratio, ctx=2048)")
axes[0].legend(loc="upper left", ncol=2, fontsize=8.8)

# Panel B: System Throughput Speedup S_TPS(C) = TPS_spec / TPS_base
s_tps_draft_fp16 = [1.436, 1.336, 1.055, 0.874, 0.981]
s_tps_eagle_fp16 = [1.045, 1.078, 0.923, 0.803, 0.869]
s_tps_ngram_fp16 = [1.120, 1.143, 0.932, 0.797, 0.823]

s_tps_draft_fp8  = [1.281, 1.269, 1.070, 0.856, 0.802]
s_tps_eagle_fp8  = [1.128, 1.119, 0.982, 0.839, 0.821]
s_tps_ngram_fp8  = [1.236, 1.174, 0.992, 0.846, 0.814]

axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Parity S(C) = 1.0")
axes[1].plot(concurrency, s_tps_draft_fp16, "o--", color="#d62728", lw=2, ms=7, label=r"Draft 1B (FP16, $C^*_{\mathrm{TPS}}=24.5$)")
axes[1].plot(concurrency, s_tps_draft_fp8,  "o-",  color="#d62728", lw=2.5, ms=8, label=r"Draft 1B (FP8, $C^*_{\mathrm{TPS}}=25.2$)")
axes[1].plot(concurrency, s_tps_eagle_fp16, "^--", color="#ff7f0e", lw=2, ms=7, label=r"EAGLE (FP16, $C^*_{\mathrm{TPS}}=8.0$)")
axes[1].plot(concurrency, s_tps_eagle_fp8,  "^-",  color="#ff7f0e", lw=2.5, ms=8, label=r"EAGLE (FP8, $C^*_{\mathrm{TPS}}=13.4$)")
axes[1].plot(concurrency, s_tps_ngram_fp16, "s--", color="#9467bd", lw=2, ms=7, label=r"N-gram (FP16, $C^*_{\mathrm{TPS}}=10.3$)")
axes[1].plot(concurrency, s_tps_ngram_fp8,  "s-",  color="#9467bd", lw=2.5, ms=8, label=r"N-gram (FP8, $C^*_{\mathrm{TPS}}=15.0$)")

axes[1].annotate("Compute Wall & Draft KV Tax\n(Disable Speculation for C > C*)",
                 xy=(64, 0.856), xytext=(24, 1.18),
                 arrowprops=dict(facecolor="red", shrink=0.06, width=1.5, headwidth=6),
                 fontweight="bold")
axes[1].set_xscale("log", base=2)
axes[1].set_xlabel("Concurrency (C)")
axes[1].set_ylabel(r"System Throughput Ratio $S_{\mathrm{TPS}}(C) = \mathrm{TPS}_{\mathrm{spec}} / \mathrm{TPS}_{\mathrm{base}}$")
axes[1].set_title("B. System Goodput Speedup (TPS Ratio, ctx=2048)")
axes[1].legend(loc="upper right", ncol=1, fontsize=8.8)

plt.tight_layout()
f2 = out_dir / "figure2_speculative_crossover.png"
plt.savefig(f2)
plt.close()

# -------------------------------------------------------------
# Figure 3: KV Pool Memory Capacity & Speculator KV Block Tax
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))

contexts = ["ctx = 512", "ctx = 2048", "ctx = 7680"]
analytic_fp16 = [276, 92, 26]
analytic_fp8  = [543, 181, 52]

x = np.arange(len(contexts))
width = 0.35
r1 = axes[0].bar(x - width/2, analytic_fp16, width, label="FP16 KV (13,249 Blocks)", color="#1f77b4")
r2 = axes[0].bar(x + width/2, analytic_fp8,  width, label="FP8 KV (26,099 Blocks)",  color="#2ca02c")
axes[0].set_ylabel("Max Analytic Concurrent Requests ($C_{\\max}$)")
axes[0].set_title("A. Analytic Concurrency Ceiling (1.97x - 2.00x Gain)")
axes[0].set_xticks(x, contexts)
axes[0].set_ylim(0, 620)
axes[0].legend()
for p in r1:
    axes[0].annotate(f"{p.get_height()}", (p.get_x() + p.get_width()/2., p.get_height()),
                     ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")
for p in r2:
    axes[0].annotate(f"{p.get_height()}", (p.get_x() + p.get_width()/2., p.get_height()),
                     ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points", color="green", weight="bold")

# Panel B: KV Block Pool Allocation across Speculators (at ctx=2048)
spec_labels = ["No Spec\n(Baseline)", "N-gram\n(Prompt Lookup)", "EAGLE\n(Feature Head)", "Draft Model\n(Llama-3.2-1B)", "Prefix Cache\n(Shared Prompt)"]
blocks_fp16 = [13248, 13210, 11869, 9618, 13641]
blocks_fp8  = [26099, 26027, 23361, 18924, 26885]

x2 = np.arange(len(spec_labels))
b1 = axes[1].bar(x2 - width/2, blocks_fp16, width, label="FP16 KV Blocks", color="#1f77b4")
b2 = axes[1].bar(x2 + width/2, blocks_fp8,  width, label="FP8 KV Blocks",  color="#2ca02c")
axes[1].set_ylabel("Allocated PagedAttention GPU Blocks")
axes[1].set_title("B. Speculator VRAM Footprint & KV Block Pool Tax")
axes[1].set_xticks(x2, spec_labels, fontsize=9.5)
axes[1].set_ylim(0, 31000)
axes[1].legend()
for p in b1:
    axes[1].annotate(f"{p.get_height():,}", (p.get_x() + p.get_width()/2., p.get_height()),
                     ha="center", va="bottom", fontsize=8.5, xytext=(0, 2), textcoords="offset points")
for p in b2:
    axes[1].annotate(f"{p.get_height():,}", (p.get_x() + p.get_width()/2., p.get_height()),
                     ha="center", va="bottom", fontsize=8.5, xytext=(0, 2), textcoords="offset points", color="green", weight="bold")

plt.tight_layout()
f3 = out_dir / "figure3_kv_capacity_gain.png"
plt.savefig(f3)
plt.close()

# -------------------------------------------------------------
# Figure 4: Quality (Perplexity) & Prefix Caching Synergy
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))

ctx_q_fp16 = [512, 2048, 4096]
ppl_fp16   = [10.8357, 8.2412, 7.6858]
ctx_q_fp8  = [512, 2048, 4096, 7680]
ppl_fp8    = [10.8548, 8.2629, 7.7034, 7.5564]

axes[0].plot(ctx_q_fp16, ppl_fp16, "o--", color="#1f77b4", lw=2.5, ms=9, label="FP16 KV (auto)")
axes[0].plot(ctx_q_fp8,  ppl_fp8,  "s-",  color="#2ca02c", lw=2.5, ms=8, label="FP8 KV (e4m3)")
axes[0].set_xlabel("Evaluation Context Length (tokens)")
axes[0].set_ylabel("WikiText-2 Perplexity (Lower is Better)")
axes[0].set_title("A. Quality Preservation: WikiText-2 Perplexity (<0.26% Delta)")
axes[0].set_xticks([512, 2048, 4096, 7680])
axes[0].legend()
for c, p16, p8 in zip(ctx_q_fp16, ppl_fp16, ppl_fp8[:3]):
    pct = 100.0 * (p8 - p16) / p16
    axes[0].annotate(f"+{pct:.2f}%", xy=(c, p8), xytext=(0, 10), textcoords="offset points",
                     ha="center", fontsize=9.5, fontweight="bold", color="#2ca02c")
axes[0].annotate("FP8 completes 7680\n(PPL = 7.5564)\nFP16 hits 8k limit", xy=(7680, 7.5564), xytext=(4800, 8.3),
                 arrowprops=dict(facecolor="green", shrink=0.08, width=1, headwidth=6), fontsize=9.5)
axes[0].text(0.04, 0.08, "HumanEval pass@1 (164 tasks):\n  • FP16: 59.76% (98 / 164)\n  • FP8:   58.54% (96 / 164)\n  • Δ = -1.22% (within ±3.85% SE)",
             transform=axes[0].transAxes, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#6c757d", alpha=0.95))

# Panel B: Prefix Caching Throughput Synergy (Stage 5 vs Stage 1/3 at ctx=2048)
c_pc = [1, 32, 256]
tps_pc_fp16 = [72.0, 1024.7, 1263.7]
tps_pc_fp8  = [73.2, 1152.1, 1905.7]
c_nopc = [1, 4, 16, 64, 256]
tps_nopc_fp16 = [70.7, 225.3, 567.1, 860.4, 779.9]
tps_nopc_fp8  = [72.0, 236.1, 608.5, 1024.2, 1136.4]

axes[1].plot(c_nopc, tps_nopc_fp16, "o:", color="#1f77b4", lw=2, ms=6, label="FP16 (No Prefix Cache)")
axes[1].plot(c_nopc, tps_nopc_fp8,  "s:", color="#2ca02c", lw=2, ms=6, label="FP8 (No Prefix Cache)")
axes[1].plot(c_pc,   tps_pc_fp16,   "o-", color="#1f77b4", lw=2.5, ms=8, label="FP16 + Prefix Caching")
axes[1].plot(c_pc,   tps_pc_fp8,    "s-", color="#2ca02c", lw=2.5, ms=8, label="FP8 + Prefix Caching")
axes[1].set_xscale("log", base=2)
axes[1].set_xlabel("Concurrency (C)")
axes[1].set_ylabel("Throughput (tokens / sec)")
axes[1].set_title("B. Prefix Caching + FP8 Synergy (ctx=2048, C=256)")
axes[1].legend(loc="upper left")
axes[1].annotate("1,905.7 tok/s\n(+50.8% vs FP16+PC\n+144% vs FP16 base)",
                 xy=(256, 1905.7), xytext=(24, 1550),
                 arrowprops=dict(facecolor="green", shrink=0.08, width=1.5, headwidth=6),
                 fontweight="bold")

plt.tight_layout()
f4 = out_dir / "figure4_quality_and_prefix_synergy.png"
plt.savefig(f4)
plt.close()

for f in [f1, f2, f3, f4]:
    shutil.copy2(f, artifact_dir / f.name)

print("All 4 publication figures generated and copied to artifacts!")
