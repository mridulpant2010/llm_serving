import sys
# usage: python evaluation/make_analysis.py [results_dir] [output_dir]
# results_dir is the merged dataset written by evaluation/merge_results.py (default: results_final)
U = (sys.argv[1] if len(sys.argv) > 1 else "results_final").rstrip("/") + "/"
OUT = (sys.argv[2] if len(sys.argv) > 2 else "evaluation/output").rstrip("/") + "/"
import json, re, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

os.makedirs(OUT, exist_ok=True)

# ---------- palette (validated reference palette; fixed entity -> colour) ----------
FP16, FP8 = "#2a78d6", "#eb6834"
DRAFT, EAGLE, NGRAM, BASE = "#6250d6", "#1baf7a", "#c98500", "#6b7280"
INK, INK2, GRID, SURF = "#0f172a", "#52514e", "#e5e7eb", "#ffffff"
MCOL = {"draft_model": DRAFT, "eagle": EAGLE, "ngram": NGRAM, "none": BASE}
MNAME = {"draft_model": "Draft model (1B)", "eagle": "EAGLE", "ngram": "N-gram", "none": "No speculation"}
plt.rcParams.update({
    "font.family": "Inter", "font.size": 15, "axes.titlesize": 18, "axes.titleweight": "bold",
    "axes.labelsize": 15, "axes.edgecolor": "#9ca3af", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "legend.frameon": False, "legend.fontsize": 14,
})
W, H = 13.33, 7.5

def save(fig, name, title, sub=None):
    fig.suptitle(title, x=0.045, y=0.985, ha="left", fontsize=22, fontweight="bold", color=INK)
    if sub:
        fig.text(0.045, 0.915, sub, ha="left", fontsize=14.5, color=INK2)
    fig.savefig(OUT + name, dpi=200)
    plt.close(fig)

# ---------- load ----------
rows = [json.loads(l) for l in open(U + "results.jsonl") if l.strip()]
df = pd.DataFrame([r for r in rows if r["status"] == "ok"])
# the original FP16 no-speculation rows exist only to pair the original N-gram sweep; plots use the re-run baseline
dfa = df[~((df.source == "original") & (df.spec == "none") & (df.stage == "stage3_spec_fp16"))]
g = dfa.groupby(["stage", "kv_dtype", "spec", "workload", "context_len", "concurrency"])
agg = g.agg(tps=("throughput_tps", "mean"), req=("req_throughput", "mean"), tpot=("tpot_ms_p50", "mean"),
            ttft=("ttft_ms_p50", "mean"), ttft99=("ttft_ms_p99", "mean"), pre=("srv_preemptions_delta", "mean"),
            runmax=("srv_running_max", "mean"), pw=("gpu_power_w_mean", "mean")).reset_index()
agg["lat"] = agg.concurrency / agg.req          # Little's law: mean request latency (s)
agg["jtok"] = agg.pw / agg.tps
s1 = agg[agg.stage == "stage1_kv_capacity"]

def S1(kv, ctx, c, col):
    return s1[(s1.kv_dtype == kv) & (s1.context_len == ctx) & (s1.concurrency == c)][col].iloc[0]

# spec speedups paired per repeat
corp = df[df.workload == "corpus"]
SPD = []
for kv in ["auto", "fp8"]:
    base = corp[(corp.kv_dtype == kv) & (corp.spec == "none")]
    for sp in ["draft_model", "eagle", "ngram"]:
        d = corp[(corp.kv_dtype == kv) & (corp.spec == sp)]
        for c in sorted(d.concurrency.unique()):
            sT, sP = [], []
            for _, r in d[d.concurrency == c].iterrows():
                b = base[(base.concurrency == c) & (base.rep == r.rep) & (base.source == r.source)].iloc[0]
                sT.append(b.tpot_ms_p50 / r.tpot_ms_p50); sP.append(r.throughput_tps / b.throughput_tps)
            SPD.append(dict(kv=kv, spec=sp, C=c, S_tpot=np.mean(sT), S_tps=np.mean(sP),
                            tps_lo=min(sP), tps_hi=max(sP), tpot_lo=min(sT), tpot_hi=max(sT)))
SPD = pd.DataFrame(SPD)
sp_ab = agg[agg.stage.isin(["stage3_spec_fp16", "stage4_spec_fp8"])]

# server-log speculative metrics + KV pool sizes
LOGS = {("auto", "draft_model"): "stage3_spec_fp16__kv-auto__spec-draft_model__pc-0.log",
        ("auto", "eagle"): "stage3_spec_fp16__kv-auto__spec-eagle__pc-0.log",
        ("auto", "ngram"): "stage3_spec_fp16__kv-auto__spec-ngram__pc-0.log",
        ("fp8", "draft_model"): "stage4_spec_fp8__kv-fp8__spec-draft_model__pc-0.log",
        ("fp8", "eagle"): "stage4_spec_fp8__kv-fp8__spec-eagle__pc-0.log",
        ("fp8", "ngram"): "stage4_spec_fp8__kv-fp8__spec-ngram__pc-0.log"}
rx = re.compile(r"Accepted: (\d+) tokens, Drafted: (\d+) tokens, Per-position acceptance rate: ([\d., ]+), Avg")
LOGM = {}
for k, f in LOGS.items():
    A = D = 0; pos = np.zeros(4); wsum = 0
    for line in open(U + "server_logs/" + f):
        m = rx.search(line)
        if m:
            a, d = int(m[1]), int(m[2]); p = np.array([float(x) for x in m[3].split(",")])
            A += a; D += d; pos += p * d; wsum += d
    LOGM[k] = dict(accepted=A, drafted=D, ratio=D / A, pos=pos / wsum, acc=A / D)
kvb = df[df.stage.isin(["stage3_spec_fp16", "stage4_spec_fp8"])].groupby(["kv_dtype", "spec", "source"]).kv_blocks.mean()
BL = {}
for (kv, sp), _ in list(LOGS.items()) + [(("auto", "none"), 0), (("fp8", "none"), 0)]:
    src = "original" if (kv, sp) == ("auto", "ngram") else "rerun"
    BL[(kv, sp)] = float(kvb[(kv, sp, src)])
# original FP16 ngram pool is compared with the original FP16 baseline pool
BL_ORIG_FP16_NONE = float(kvb[("auto", "none", "original")])

def lab(ax, x, y, t, dy=0.0, **kw):
    ax.text(x, y + dy, t, ha=kw.pop("ha", "center"), va=kw.pop("va", "bottom"), fontsize=kw.pop("fontsize", 13), **kw)

# =====================================================================================
# 1. Prefill ceiling
# =====================================================================================
fig, ax = plt.subplots(figsize=(W, H)); fig.subplots_adjust(left=.08, right=.97, top=.80, bottom=.13)
ctxs = [512, 2048, 7680]; x = np.arange(3); bw = .26
ceil = [256 / (S1("auto", c, 1, "ttft") / 1e3) for c in ctxs]
p16 = [s1[(s1.kv_dtype == "auto") & (s1.context_len == c)].tps.max() for c in ctxs]
p8 = [s1[(s1.kv_dtype == "fp8") & (s1.context_len == c)].tps.max() for c in ctxs]
ax.bar(x - bw, ceil, bw * .95, facecolor="white", edgecolor="#9ca3af", hatch="////", linewidth=1.2)
ax.bar(x, p16, bw * .95, color=FP16); ax.bar(x + bw, p8, bw * .95, color=FP8)
for i in range(3):
    lab(ax, x[i] - bw, ceil[i], f"{ceil[i]:,.0f}", 40, color=INK2)
    lab(ax, x[i], p16[i], f"{p16[i]:,.0f}\n{p16[i]/ceil[i]*100:.0f}% of\nceiling", 40, fontweight="bold", fontsize=12.5)
    lab(ax, x[i] + bw, p8[i], f"{p8[i]:,.0f}\n{p8[i]/ceil[i]*100:.0f}% of\nceiling", 40, fontweight="bold", fontsize=12.5)
    lab(ax, x[i] + bw / 2, -460, f"FP8 gain  +{(p8[i]/p16[i]-1)*100:.0f}%", fontsize=14, fontweight="bold", color=FP8)
ax.set_xticks(x); ax.set_xticklabels([f"Context {c:,}" for c in ctxs]); ax.set_ylabel("Peak output throughput (tokens/s)")
ax.tick_params(axis="x", pad=34); ax.set_ylim(0, 6000)
ax.legend(handles=[Patch(facecolor="white", edgecolor="#9ca3af", hatch="////", label="Prefill-bound ceiling = 256 ÷ TTFT(C=1)"),
                   Patch(color=FP16, label="FP16 KV, measured peak"), Patch(color=FP8, label="FP8 KV, measured peak")], loc="upper right")
save(fig, "01_prefill_ceiling.png", "Longer prompts push serving toward a prefill-bound ceiling",
     "FP8's 2× KV capacity buys +20% to +37% peak throughput because prefill compute is untouched · Llama-3-8B, 256 output tokens")

# =====================================================================================
# 2. FP8 under load (small multiples)
# =====================================================================================
fig, axs = plt.subplots(2, 3, figsize=(W, H)); fig.subplots_adjust(left=.07, right=.98, top=.80, bottom=.10, hspace=.62, wspace=.38)
mets = [("tps", "Throughput (tok/s)\nhigher is better", 1, "{:,.0f}"), ("lat", "Mean request latency (s)\nlower is better", 1, "{:,.0f}"),
        ("ttft", "p50 time-to-first-token (s)\nlower is better", 1e-3, "{:,.0f}"), ("tpot", "p50 time-per-output-token (ms)\nlower is better", 1, "{:,.0f}"),
        ("jtok", "Energy per token (J)\nlower is better", 1, "{:.2f}"), ("pre", "Preemptions\nlower is better", 1, "{:.0f}")]
for ax, (m, t, sc, fmt) in zip(axs.flat, mets):
    for gi, ctx in enumerate([2048, 7680]):
        a = S1("auto", ctx, 256, m) * sc; b = S1("fp8", ctx, 256, m) * sc
        ax.bar(gi - .19, a, .36, color=FP16); ax.bar(gi + .19, b, .36, color=FP8)
        top = max(a, b, 1e-9)
        ax.text(gi - .19, a, fmt.format(a), ha="center", va="bottom", fontsize=12.5); ax.text(gi + .19, b, fmt.format(b), ha="center", va="bottom", fontsize=12.5)
        if a > 0:
            d = (b / a - 1) * 100
            ax.text(gi, .985, f"FP8 {d:+.0f}%", ha="center", va="top", fontsize=13, fontweight="bold", color=FP8, transform=ax.get_xaxis_transform())
        else:
            ax.text(gi, .985, "—", ha="center", va="top", fontsize=13, transform=ax.get_xaxis_transform())
    ax.set_xticks([0, 1]); ax.set_xticklabels(["ctx 2,048", "ctx 7,680"], fontsize=13)
    ax.set_title(t, fontsize=13.5, loc="left", color=INK2, fontweight="normal"); ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, max(S1("auto", 2048, 256, m), S1("fp8", 2048, 256, m), S1("auto", 7680, 256, m), S1("fp8", 7680, 256, m)) * sc * 1.45 + 1e-9)
axs[1, 2].legend(handles=[Patch(color=FP16, label="FP16 KV"), Patch(color=FP8, label="FP8 KV")], loc="upper right", bbox_to_anchor=(1.0, .86), fontsize=13)
save(fig, "02_fp8_under_load.png", "FP8 wins on throughput, queueing and energy; it loses on per-token latency",
     "Concurrency 256 · a bigger FP8 batch (≈2× requests running) slows each token but clears the queue sooner")

# =====================================================================================
# 3. TPOT vs throughput at C=256
# =====================================================================================
fig, axs = plt.subplots(1, 2, figsize=(W, H), sharey=True); fig.subplots_adjust(left=.07, right=.98, top=.80, bottom=.14, wspace=.08)
meths = ["draft_model", "eagle", "ngram"]
for ax, kv, nm in zip(axs, ["auto", "fp8"], ["FP16 KV cache", "FP8 KV cache"]):
    for i, sp in enumerate(meths):
        r = SPD[(SPD.kv == kv) & (SPD.spec == sp) & (SPD.C == 256)].iloc[0]
        ax.bar(i - .2, r.S_tpot, .36, facecolor="white", edgecolor=MCOL[sp], hatch="////", linewidth=1.6)
        ax.bar(i + .2, r.S_tps, .36, color=MCOL[sp])
        ax.text(i - .2, r.S_tpot + .03, f"{r.S_tpot:.2f}×", ha="center", fontsize=13.5, fontweight="bold")
        ax.text(i + .2, r.S_tps + .03, f"{r.S_tps:.2f}×", ha="center", fontsize=13.5, fontweight="bold")
    ax.axhline(1, color=INK, ls="--", lw=1.3); ax.set_xticks(range(3)); ax.set_xticklabels([MNAME[m] for m in meths], fontsize=14)
    ax.set_title(nm, loc="left"); ax.set_ylim(0, 2.45)
axs[0].set_ylabel("Speculation speedup vs. no speculation (×)")
axs[0].text(2.55, 1.04, "parity", fontsize=12, color=INK2, ha="right")
fig.legend(handles=[Patch(facecolor="white", edgecolor=INK2, hatch="////", label="Per-token latency (TPOT) speedup"), Patch(color=INK2, label="System throughput (tok/s) speedup")],
           loc="center right", bbox_to_anchor=(.985, .86), ncol=2, fontsize=13)
save(fig, "03_tpot_vs_throughput_c256.png", "At C=256, TPOT can say speculation wins while throughput says it loses",
     "Speculator KV blocks shrink the pool, so fewer requests run at once: lower TPOT is a smaller-batch effect")

# =====================================================================================
# 4. Throughput speedup vs C (with repeat range)
# =====================================================================================
fig, axs = plt.subplots(1, 2, figsize=(W, H), sharey=True); fig.subplots_adjust(left=.07, right=.98, top=.80, bottom=.14, wspace=.07)
Cs = [1, 4, 16, 64, 256]
for ax, kv, nm in zip(axs, ["auto", "fp8"], ["FP16 KV cache", "FP8 KV cache"]):
    ax.axhspan(0.7, 1, color="#f3f4f6", zorder=0); ax.axhline(1, color=INK, ls="--", lw=1.3)
    for sp in meths:
        d = SPD[(SPD.kv == kv) & (SPD.spec == sp)].sort_values("C")
        ax.fill_between(d.C, d.tps_lo, d.tps_hi, color=MCOL[sp], alpha=.18, lw=0)
        ax.plot(d.C, d.S_tps, "-o", color=MCOL[sp], lw=2.4, ms=7, mec="white", mew=1.5)
    ax.set_xscale("log", base=4); ax.set_xticks(Cs); ax.set_xticklabels(Cs); ax.minorticks_off(); ax.set_xlim(.8, 280)
    ax.set_ylim(.7, 1.6); ax.set_title(nm, loc="left"); ax.set_xlabel("Concurrency (C)")
    ax.text(1.0, .715, "speculation loses throughput", fontsize=12.5, color=INK2)
axs[0].set_ylabel("Throughput speedup (×)")
fig.legend(handles=[Line2D([], [], color=MCOL[m], lw=3, marker="o", label=MNAME[m]) for m in meths], loc="center right", bbox_to_anchor=(.985, .86), ncol=3, fontsize=13)
save(fig, "04_throughput_speedup_vs_C.png", "Speculation pays only at low load, and FP8 weakens the draft model's gain",
     "Band = min/max of 3 repeats (2 for FP16 N-gram) · five concurrency points, so no crossover C* is interpolated")

# =====================================================================================
# 5. Absolute throughput by config + winner strip
# =====================================================================================
cfg = []
for kv in ["auto", "fp8"]:
    for sp in ["none", "draft_model", "eagle", "ngram"]:
        d = sp_ab[(sp_ab.kv_dtype == kv) & (sp_ab.spec == sp)].sort_values("concurrency")
        cfg.append((kv, sp, d))
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([.08, .32, .74, .48])
for kv, sp, d in cfg:
    emph = sp in ("none", "draft_model")
    ax.plot(d.concurrency, d.tps, "-o" if kv == "fp8" else "--o", color=MCOL[sp] if sp != "none" else (FP8 if kv == "fp8" else FP16),
            lw=3 if emph else 1.6, ms=7 if emph else 4, alpha=1 if emph else .35, mec="white", mew=1.2)
ax.set_xscale("log", base=4); ax.set_xticks(Cs); ax.set_xticklabels(Cs); ax.minorticks_off(); ax.set_xlim(.8, 300)
ax.set_xlabel("Concurrency (C)"); ax.set_ylabel("Output throughput (tok/s)")
def endlab(kv, sp, text, col, dy=0):
    d = [c for c in cfg if c[0] == kv and c[1] == sp][0][2]; ax.text(290, d.tps.iloc[-1] + dy, text, color=col, fontsize=13, va="center", fontweight="bold")
endlab("fp8", "none", "FP8, no spec", FP8, 40); endlab("fp8", "draft_model", "FP8 + draft", DRAFT, -45)
endlab("auto", "none", "FP16, no spec", FP16, 32); endlab("auto", "draft_model", "FP16 + draft", DRAFT, -32)
ax.legend(handles=[Line2D([], [], color=INK2, lw=3, label="solid = FP8 KV"), Line2D([], [], color=INK2, lw=2, ls="--", label="dashed = FP16 KV"),
                   Line2D([], [], color=INK2, lw=1.4, alpha=.4, label="faint = EAGLE, N-gram")], loc="upper left", fontsize=13)
# winner strip
ax2 = fig.add_axes([.08, .04, .74, .12]); ax2.set_xscale("log", base=4); ax2.set_xlim(.8, 300); ax2.set_ylim(0, 1); ax2.axis("off")
fig.text(.08, .215, "Best configuration at each C (highest tok/s)", fontsize=13.5, color=INK2, fontweight="bold")
winners = []
for c in Cs:
    best = max(((kv, sp, float(d[d.concurrency == c].tps.iloc[0])) for kv, sp, d in cfg), key=lambda t: t[2]); winners.append((c, best))
    kvn = "FP8" if best[0] == "fp8" else "FP16"; spn = {"none": "no spec", "draft_model": "+ draft", "eagle": "+ EAGLE", "ngram": "+ N-gram"}[best[1]]
    col = MCOL[best[1]] if best[1] != "none" else (FP8 if best[0] == "fp8" else FP16)
    ax2.text(c, .35, f"{kvn}\n{spn}", ha="center", va="center", fontsize=13, fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=.35", fc=col, ec="none"))
save(fig, "05_throughput_by_config_and_winner.png", "Speculation wins only at C ≤ 4 (FP16 draft model); FP8 alone wins from C=16",
     "Absolute throughput by configuration, ctx 2,048 · clean re-run, 3 repeats (FP16 N-gram from the original run, 2 repeats)")

# =====================================================================================
# 6. Speculator tax: KV pool + verification waste + per-position acceptance
# =====================================================================================
fig, axs = plt.subplots(1, 3, figsize=(W, H)); fig.subplots_adjust(left=.06, right=.985, top=.80, bottom=.14, wspace=.32)
ax = axs[0]
for i, sp in enumerate(meths):
    for j, (kv, col) in enumerate([("auto", FP16), ("fp8", FP8)]):
        v = BL[(kv, sp)] / BL[(kv, "none")] * 100
        ax.bar(i + (j - .5) * .38, v, .36, color=col)
        if v < 99: ax.text(i + (j - .5) * .38, v + 1, f"{v:.0f}%", ha="center", fontsize=11.5, fontweight="bold")
        elif j == 0: ax.text(i, v + 2, "≈100%", ha="center", fontsize=11.5, fontweight="bold")
ax.set_ylim(0, 125); ax.set_xticks(range(3)); ax.set_xticklabels(["Draft", "EAGLE", "N-gram"]); ax.axhline(100, color=INK, ls="--", lw=1.1)
ax.set_title("KV blocks left vs.\nno speculation", fontsize=15, loc="left")
ax = axs[1]
for i, sp in enumerate(meths):
    for j, (kv, col) in enumerate([("auto", FP16), ("fp8", FP8)]):
        v = LOGM[(kv, sp)]["ratio"]; ax.bar(i + (j - .5) * .38, v, .36, color=col); ax.text(i + (j - .5) * .38, v + .12, f"{v:.1f}", ha="center", fontsize=12.5, fontweight="bold")
ax.set_ylim(0, 10.5); ax.set_xticks(range(3)); ax.set_xticklabels(["Draft", "EAGLE", "N-gram"]); ax.axhline(1, color=INK, ls="--", lw=1.1)
ax.set_title("Drafted tokens verified per\ntoken accepted", fontsize=15, loc="left")
ax = axs[2]
for sp in meths:
    p = LOGM[("auto", sp)]["pos"]; ax.plot([1, 2, 3, 4], p, "-o", color=MCOL[sp], lw=2.6, ms=7, mec="white", mew=1.4)
    ax.text(4.12, p[-1], MNAME[sp].split(" ")[0], color=MCOL[sp], fontsize=13, va="center", fontweight="bold")
ax.set_xticks([1, 2, 3, 4]); ax.set_xlim(.8, 5.1); ax.set_ylim(0, .85); ax.set_xlabel("Draft position"); ax.set_title("Acceptance rate by\ndraft position (FP16)", fontsize=15, loc="left")
axs[0].legend(handles=[Patch(color=FP16, label="FP16 KV"), Patch(color=FP8, label="FP8 KV")], loc="upper left", ncol=2, fontsize=11.5, columnspacing=1, handlelength=1.2)
save(fig, "06_speculator_tax.png", "Speculation has two hidden costs: lost KV memory and wasted verification",
     "EAGLE verifies ≈6 tokens per useful one; only its first draft token is often accepted · server-log counters, whole sweep")

# =====================================================================================
# 7. Energy per token
# =====================================================================================
fig, axs = plt.subplots(1, 3, figsize=(W, H), sharey=True); fig.subplots_adjust(left=.07, right=.985, top=.80, bottom=.15, wspace=.1)
for ax, ctx in zip(axs, [512, 2048, 7680]):
    for kv, col in [("auto", FP16), ("fp8", FP8)]:
        d = s1[(s1.kv_dtype == kv) & (s1.context_len == ctx)].sort_values("concurrency")
        ax.plot(d.concurrency, d.jtok, "-o", color=col, lw=2.6, ms=6, mec="white", mew=1.3)
    a, b = S1("auto", ctx, 256, "jtok"), S1("fp8", ctx, 256, "jtok")
    ax.annotate(f"FP8 −{(1-b/a)*100:.0f}%\nat C=256", xy=(256, b), xytext=(1.15, .13), fontsize=13.5, fontweight="bold", color=FP8,
                arrowprops=dict(arrowstyle="-", color=FP8, lw=1.2))
    ax.set_xscale("log", base=4); ax.set_yscale("log"); ax.set_ylim(.07, 7); ax.set_xticks(Cs); ax.set_xticklabels(Cs); ax.minorticks_off()
    ax.set_title(f"Context {ctx:,}", loc="left"); ax.set_xlabel("Concurrency (C)")
axs[0].set_ylabel("Energy per output token (J, log scale)")
axs[0].set_yticks([.1, .3, 1, 3]); axs[0].set_yticklabels(["0.1", "0.3", "1", "3"])
axs[0].legend(handles=[Line2D([], [], color=FP16, lw=3, label="FP16 KV"), Line2D([], [], color=FP8, lw=3, label="FP8 KV")], loc="upper right", fontsize=13)
save(fig, "07_energy_per_token.png", "GPU power is pinned near 300 W, so energy per token tracks throughput",
     "Batching cuts energy per token up to ~40× (ctx 512, C=1 to 256); FP8 saves another 17–31% at saturation")

# =====================================================================================
# 8. Model validation
# =====================================================================================
Wg = 8.03e9 * 2 / 1e9; KVT = 2 * 32 * 8 * 128 * 2 / 1e9
bw = Wg / (S1("auto", 512, 1, "tpot") / 1e3)
fig, axs = plt.subplots(1, 2, figsize=(W, H)); fig.subplots_adjust(left=.07, right=.985, top=.80, bottom=.15, wspace=.25)
ax = axs[0]
pts = []
for kv, f, col in [("auto", 1, FP16), ("fp8", .5, FP8)]:
    for ctx in [2048, 7680]:
        pred = (Wg + (ctx + 128) * KVT * f) / bw * 1e3; mea = S1(kv, ctx, 1, "tpot")
        ax.scatter(pred, mea, s=170, color=col, zorder=3, edgecolor="white", linewidth=1.5)
        ax.annotate(f"ctx {ctx:,}", (pred, mea), xytext=(8, -18 if ctx == 2048 else 10), textcoords="offset points", fontsize=12.5, color=INK2)
lo, hi = 13.2, 14.4
ax.plot([lo, hi], [lo, hi], color=INK, ls="--", lw=1.3); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.text(13.95, 13.88, "perfect\nagreement", fontsize=12.5, color=INK2, rotation=0)
ax.set_xlabel("Model-predicted TPOT (ms)"); ax.set_ylabel("Measured TPOT at C=1 (ms)")
ax.set_title("Single-stream decode matches\nthe bandwidth model", loc="left", fontsize=15)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=FP16, ms=11, label="FP16 KV"), Line2D([], [], marker="o", ls="", color=FP8, ms=11, label="FP8 KV")], loc="upper left", fontsize=13)
ax = axs[1]; k = 0; lbls = []
for ctx in [2048, 7680]:
    for kv, col in [("auto", FP16), ("fp8", FP8)]:
        an = int(df[(df.stage == "stage1_kv_capacity") & (df.kv_dtype == kv) & (df.context_len == ctx)].max_concurrency_analytic.iloc[0])
        ob = S1(kv, ctx, 256, "runmax")
        ax.bar(k - .19, an, .36, facecolor="white", edgecolor="#6b7280", hatch="////", lw=1.3); ax.bar(k + .19, ob, .36, color=col)
        ax.text(k - .19, an + 3, f"{an}", ha="center", fontsize=12); ax.text(k + .19, ob + 3, f"{ob:.0f}", ha="center", fontsize=12, fontweight="bold")
        lbls.append(f"{'FP16' if kv=='auto' else 'FP8'}\nctx {ctx:,}"); k += 1
ax.set_xticks(range(4)); ax.set_xticklabels(lbls, fontsize=13); ax.set_ylim(0, 225); ax.set_ylabel("Concurrent requests")
ax.set_title("Analytic KV capacity matches the\nmax requests vLLM actually ran", loc="left", fontsize=15)
ax.legend(handles=[Patch(facecolor="white", edgecolor="#6b7280", hatch="////", label="Analytic capacity"), Patch(color=INK2, label="Observed running max (C=256)")], loc="upper right", fontsize=12.5)
save(fig, "08_model_validation.png", "Two first-principles models hold up against the measurements",
     "Decode is weight-streaming bound (≈1.2 TB/s effective); observed capacity is within 9% of the KV-pool prediction")

# =====================================================================================
# 9. Prefix caching (with the cache-off control from the clean re-run)
# =====================================================================================
fig, axs = plt.subplots(1, 2, figsize=(W, H), gridspec_kw=dict(width_ratios=[1.35, 1])); fig.subplots_adjust(left=.07, right=.985, top=.80, bottom=.17, wspace=.28)
pcon = agg[agg.stage == "stage5_prefix_on"]; pcoff = agg[agg.stage == "stage5_cache_off"]
def P(d, kv, c): return float(d[(d.kv_dtype == kv) & (d.concurrency == c)].tps.iloc[0])
ax = axs[0]
bars = [(P(pcoff, "auto", 256), FP16, True, "FP16\ncache off"), (P(pcon, "auto", 256), FP16, False, "FP16\ncache on"),
        (P(pcoff, "fp8", 256), FP8, True, "FP8\ncache off"), (P(pcon, "fp8", 256), FP8, False, "FP8\ncache on")]
for i, (v, col, off, t) in enumerate(bars):
    x = i + (0.3 if i >= 2 else 0)
    ax.bar(x, v, .62, facecolor="white" if off else col, edgecolor=col, hatch="////" if off else None, lw=1.8)
    ax.text(x, v + 30, f"{v:,.0f}", ha="center", fontsize=15, fontweight="bold")
ax.set_xticks([0, 1, 2.3, 3.3]); ax.set_xticklabels([b[3] for b in bars], fontsize=13.5)
ax.set_ylabel("Throughput at C=256, ctx 2,048 (tok/s)"); ax.set_ylim(0, 2300)
g16 = P(pcon, "auto", 256) / P(pcoff, "auto", 256) - 1; g8 = P(pcon, "fp8", 256) / P(pcoff, "fp8", 256) - 1
ax.text(.5, 2130, f"cache gain {g16*100:+.0f}%", ha="center", fontsize=14, fontweight="bold", color=FP16)
ax.text(2.8, 2130, f"cache gain {g8*100:+.0f}%", ha="center", fontsize=14, fontweight="bold", color=FP8)
ax.set_title("Same shared-prefix workload, cache off vs on", loc="left", fontsize=15)
ax = axs[1]
cs = [1, 32, 256]
for kv, col in [("auto", FP16), ("fp8", FP8)]:
    ax.plot(range(3), [P(pcon, kv, c) / P(pcoff, kv, c) for c in cs], "-o", color=col, lw=2.6, ms=8, mec="white", mew=1.4)
    for k, c in enumerate(cs):
        ax.text(k, P(pcon, kv, c) / P(pcoff, kv, c) + (.04 if kv == "fp8" else -.06), f"{P(pcon, kv, c) / P(pcoff, kv, c):.2f}×", ha="center", fontsize=12.5, color=col, fontweight="bold", va="bottom" if kv == "fp8" else "top")
ax.axhline(1, color=INK, ls="--", lw=1.2); ax.set_xticks(range(3)); ax.set_xticklabels(cs); ax.set_xlabel("Concurrency (C)"); ax.set_ylim(.85, 1.95)
ax.set_ylabel("Cache-on / cache-off throughput (×)"); ax.set_title("The cache helps more under load", loc="left", fontsize=15)
ax.legend(handles=[Line2D([], [], color=FP16, lw=3, label="FP16 KV"), Line2D([], [], color=FP8, lw=3, label="FP8 KV")], loc="upper left", fontsize=13)
fv = P(pcon, "fp8", 256) / P(pcon, "auto", 256) - 1; fo = P(pcoff, "fp8", 256) / P(pcoff, "auto", 256) - 1
save(fig, "09_prefix_caching.png", "Prefix caching adds about +65% for both dtypes; FP8's own gain is mostly capacity",
     f"FP8 over FP16 at C=256: {fo*100:+.0f}% with the cache off, {fv*100:+.0f}% with it on · 3 repeats, clean re-run")

# =====================================================================================
# 10. Quality
# =====================================================================================
q = pd.DataFrame([json.loads(l) for l in open(U + "quality.jsonl")])
fig, axs = plt.subplots(1, 3, figsize=(W, H)); fig.subplots_adjust(left=.07, right=.985, top=.80, bottom=.14, wspace=.38)
ax = axs[0]
for kv, col in [("auto", FP16), ("fp8", FP8)]:
    d = q[q.kv_dtype == kv].sort_values("context_len"); ax.plot(range(len(d)), d.ppl, "-o", color=col, lw=2.6, ms=8, mec="white", mew=1.4)
d8 = q[q.kv_dtype == "fp8"].sort_values("context_len")
ax.set_xticks(range(4)); ax.set_xticklabels(["512", "2k", "4k", "7.7k"]); ax.set_xlabel("Evaluation context (tokens)"); ax.set_ylabel("WikiText-2 perplexity (lower is better)")
ax.text(3, d8.ppl.iloc[-1] + .5, "FP8 only", ha="center", fontsize=12.5, color=INK2)
ax.legend(handles=[Line2D([], [], color=FP16, lw=3, label="FP16 KV"), Line2D([], [], color=FP8, lw=3, label="FP8 KV")], loc="upper right", fontsize=13)
ax.set_title("Perplexity by context", loc="left", fontsize=15)
ax = axs[1]
pa = q[q.kv_dtype == "auto"].set_index("context_len").ppl; pf = q[q.kv_dtype == "fp8"].set_index("context_len").ppl
for i, c in enumerate([512, 2048, 4096]):
    v = (pf[c] / pa[c] - 1) * 100; ax.bar(i, v, .55, color=FP8); ax.text(i, v + .008, f"+{v:.2f}%", ha="center", fontsize=14, fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(["512", "2k", "4k"]); ax.set_ylim(0, .4); ax.set_ylabel("Perplexity increase, FP8 vs FP16 (%)"); ax.set_xlabel("Evaluation context (tokens)")
ax.set_title("Relative perplexity change", loc="left", fontsize=15)
ax = axs[2]
he = {"FP16 KV": (0.5975609756097561, 0.03841026959095088, FP16, 98), "FP8 KV": (0.5853658536585366, 0.03858801357403451, FP8, 96)}
for i, (nm, (v, se, col, k)) in enumerate(he.items()):
    ax.bar(i, v * 100, .55, color=col); ax.errorbar(i, v * 100, yerr=se * 100, color=INK, capsize=7, lw=2)
    ax.text(i, 3, f"{v*100:.1f}%\n({k}/164)", ha="center", color="white", fontsize=14, fontweight="bold")
ax.set_xticks(range(2)); ax.set_xticklabels(list(he)); ax.set_ylim(0, 80); ax.set_ylabel("HumanEval pass@1 (%)")
ax.text(.5, 70, "−1.2 pts = 2 of 164 problems,\nwithin 1 standard error", ha="center", fontsize=12.5, color=INK2)
ax.set_title("Code generation (HumanEval)", loc="left", fontsize=15)
save(fig, "10_quality_fp8.png", "FP8 KV cache: tiny quality cost on perplexity and HumanEval",
     "Perplexity +0.18% to +0.26% · HumanEval −1.2 pts (error bars = 1 standard error, n=164) · MMLU not yet run")

# =====================================================================================
# comparison tables (xlsx + csv)
# =====================================================================================
t1 = s1[["kv_dtype", "context_len", "concurrency", "tps", "tpot", "ttft", "lat", "pre", "runmax", "jtok"]].copy()
t1.columns = ["KV dtype", "Context", "C", "Throughput (tok/s)", "TPOT p50 (ms)", "TTFT p50 (ms)", "Mean request latency (s)", "Preemptions", "Max running", "Energy (J/token)"]
t1["KV dtype"] = t1["KV dtype"].map({"auto": "FP16", "fp8": "FP8"})
t2 = SPD.copy(); t2["kv"] = t2.kv.map({"auto": "FP16", "fp8": "FP8"}); t2["spec"] = t2.spec.map(MNAME)
t2 = t2.rename(columns={"kv": "KV dtype", "spec": "Method", "S_tpot": "Speedup TPOT (x)", "S_tps": "Speedup throughput (x)", "tps_lo": "Throughput speedup min", "tps_hi": "Throughput speedup max", "tpot_lo": "TPOT speedup min", "tpot_hi": "TPOT speedup max"})
t3 = pd.DataFrame([dict(C=c, best=f"{'FP8' if b[0]=='fp8' else 'FP16'} {MNAME[b[1]]}", tps=b[2],
                        fp8_baseline=float(sp_ab[(sp_ab.kv_dtype == 'fp8') & (sp_ab.spec == 'none') & (sp_ab.concurrency == c)].tps.iloc[0])) for c, b in winners])
t3["gain_vs_FP8_baseline_pct"] = (t3.tps / t3.fp8_baseline - 1) * 100
t4 = pd.DataFrame([dict(KV="FP16" if kv == "auto" else "FP8", Method=MNAME[sp], KV_blocks=BL[(kv, sp)], pool_vs_no_spec_pct=BL[(kv, sp)] / BL[(kv, "none")] * 100,
                        accepted=LOGM[(kv, sp)]["accepted"], drafted=LOGM[(kv, sp)]["drafted"], drafted_per_accepted=LOGM[(kv, sp)]["ratio"],
                        accept_rate=LOGM[(kv, sp)]["acc"], pos1=LOGM[(kv, sp)]["pos"][0], pos2=LOGM[(kv, sp)]["pos"][1], pos3=LOGM[(kv, sp)]["pos"][2], pos4=LOGM[(kv, sp)]["pos"][3])
                   for (kv, sp) in LOGS])
t5 = pd.DataFrame([dict(KV="FP16" if kv == "auto" else "FP8", C=c, cache_off=P(pcoff, kv, c), cache_on=P(pcon, kv, c), gain_pct=(P(pcon, kv, c) / P(pcoff, kv, c) - 1) * 100,
                        preempt_off=float(pcoff[(pcoff.kv_dtype == kv) & (pcoff.concurrency == c)].pre.iloc[0]), preempt_on=float(pcon[(pcon.kv_dtype == kv) & (pcon.concurrency == c)].pre.iloc[0]))
                   for kv in ["auto", "fp8"] for c in [1, 32, 256]])
t6 = pd.read_csv(U + "rebuilt_vs_rerun.csv") if os.path.exists(U + "rebuilt_vs_rerun.csv") else pd.DataFrame()
with pd.ExcelWriter(OUT + "comparison_tables.xlsx") as xw:
    t5.round(2).to_excel(xw, "Prefix_cache_on_vs_off", index=False)
    if len(t6): t6.to_excel(xw, "Rebuilt_vs_rerun", index=False)
    t1.round(3).to_excel(xw, "Stage1_FP16_vs_FP8", index=False); t2.round(3).to_excel(xw, "Speculation_speedups", index=False)
    t3.round(2).to_excel(xw, "Best_config_by_C", index=False); t4.round(3).to_excel(xw, "Speculator_tax", index=False)
print(t3.round(1).to_string()); print(t4.round(2).to_string())
print(open(OUT + "comparison_tables.xlsx", "rb").read()[:4])
