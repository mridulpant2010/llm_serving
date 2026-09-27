"""
Plotting utilities for KV Cache experiments.

Generates publication-ready matplotlib figures for the project report.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def plot_cache_size_comparison(context_lengths_results: list[dict], save_path: str = None):
    """
    Plot KV cache size (MB) vs context length for FP32 and INT8.

    This directly visualises the Topic 2 question:
    "How does the benefit change with context length?"
    """
    ctx_lens = [r["actual_ctx_len"] for r in context_lengths_results]
    fp32_sizes = [r["cache_size_fp32_mb"] for r in context_lengths_results]
    int8_sizes = [r["cache_size_int8_mb"] for r in context_lengths_results]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ctx_lens, fp32_sizes, "o-", color="#d32f2f", linewidth=2, markersize=8, label="FP32 / FP16 KV Cache")
    ax.plot(ctx_lens, int8_sizes, "s-", color="#1976d2", linewidth=2, markersize=8, label="INT8 / FP8 KV Cache")

    # Shade the savings area
    ax.fill_between(ctx_lens, int8_sizes, fp32_sizes, alpha=0.15, color="#1976d2", label="Memory saved")

    ax.set_xlabel("Context Length (tokens)", fontsize=13)
    ax.set_ylabel("KV Cache Size (MB)", fontsize=13)
    ax.set_title("KV Cache Memory Footprint vs Context Length", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()
    return fig


def plot_concurrency_throughput(batch_results: list[dict], save_path: str = None):
    """
    Plot throughput (tokens/sec) vs batch size (simulated concurrency).

    This answers: "Does extra concurrency from FP8 turn into real throughput,
    or is it lost to a compute bottleneck?"
    """
    ok_results = [r for r in batch_results if r["status"] == "ok"]
    batch_sizes = [r["batch_size"] for r in ok_results]
    throughputs = [r["throughput_tps"] for r in ok_results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_throughput = "#2e7d32"
    ax1.plot(batch_sizes, throughputs, "o-", color=color_throughput, linewidth=2, markersize=8, label="Throughput")
    ax1.set_xlabel("Batch Size (Simulated Concurrent Users)", fontsize=13)
    ax1.set_ylabel("Throughput (tokens/sec)", fontsize=13, color=color_throughput)
    ax1.tick_params(axis="y", labelcolor=color_throughput)

    # If we have memory data, plot it on a second y-axis
    memory_data = [r.get("peak_memory_mb") for r in ok_results]
    if any(m is not None for m in memory_data):
        ax2 = ax1.twinx()
        color_memory = "#e65100"
        valid_mem = [(bs, m) for bs, m in zip(batch_sizes, memory_data) if m is not None]
        if valid_mem:
            ax2.plot(
                [x[0] for x in valid_mem],
                [x[1] for x in valid_mem],
                "^--", color=color_memory, linewidth=2, markersize=8, label="Peak Memory"
            )
            ax2.set_ylabel("Peak GPU Memory (MB)", fontsize=13, color=color_memory)
            ax2.tick_params(axis="y", labelcolor=color_memory)

    # Mark OOM point if it exists
    oom_results = [r for r in batch_results if r["status"] == "OOM"]
    if oom_results:
        oom_bs = oom_results[0]["batch_size"]
        ax1.axvline(x=oom_bs, color="#d32f2f", linestyle="--", linewidth=2, alpha=0.7)
        ax1.annotate(
            f"OOM at batch={oom_bs}",
            xy=(oom_bs, max(throughputs) * 0.9),
            fontsize=11, color="#d32f2f", fontweight="bold",
            ha="right",
        )

    ax1.set_title("Throughput vs Concurrency (Batch Size)", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()
    return fig


def plot_quantization_error(error_results: list[dict], dtype_label: str = "INT8", save_path: str = None):
    """
    Plot per-layer quantization error (MSE) to show where precision
    loss is concentrated.

    This helps answer: "Where does quality degradation become visible?"
    """
    layers = [r["layer"] for r in error_results]
    key_mse = [r["key_mse"] for r in error_results]
    val_mse = [r["value_mse"] for r in error_results]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(layers))
    width = 0.35

    ax.bar(x - width / 2, key_mse, width, label="Key MSE", color="#1976d2", alpha=0.8)
    ax.bar(x + width / 2, val_mse, width, label="Value MSE", color="#d32f2f", alpha=0.8)

    ax.set_xlabel("Transformer Layer", fontsize=13)
    ax.set_ylabel("Mean Squared Error", fontsize=13)
    ax.set_title(f"Per-Layer Quantization Error ({dtype_label} vs Original)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()
    return fig


def plot_precision_memory_bar(comparisons: dict[str, "PrecisionComparison"], save_path: str = None):
    """
    Bar chart comparing memory footprint across different precisions.

    Pass a dict like {"FP16": comparison_fp16, "INT8": comparison_int8}.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["Original\n(FP32)"]
    sizes = [list(comparisons.values())[0].original_size_mb]
    colors = ["#78909c"]

    for name, comp in comparisons.items():
        labels.append(f"Compressed\n({name})")
        sizes.append(comp.compressed_size_mb)
        colors.append("#1976d2" if "8" in name else "#2e7d32")

    bars = ax.bar(labels, sizes, color=colors, edgecolor="white", linewidth=1.5)

    # Add value labels on bars
    for bar, size in zip(bars, sizes):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"{size:.3f} MB",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("KV Cache Size (MB)", fontsize=13)
    ax.set_title("KV Cache Memory by Precision Format", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()
    return fig


def plot_latency_comparison(results: list["GenerationResult"], labels: list[str] = None, save_path: str = None):
    """
    Compare TPOT (Time Per Output Token) across different configurations.
    """
    if labels is None:
        labels = [f"Config {i}" for i in range(len(results))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # TPOT comparison
    tpots = [r.time_per_output_token_ms for r in results]
    bars1 = ax1.bar(labels, tpots, color="#1976d2", edgecolor="white")
    for bar, val in zip(bars1, tpots):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.1f}ms", ha="center", va="bottom", fontsize=11)
    ax1.set_ylabel("Time Per Output Token (ms)", fontsize=12)
    ax1.set_title("TPOT Comparison", fontsize=13, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3)

    # Tokens/sec comparison
    tps_vals = [r.tokens_per_sec for r in results]
    bars2 = ax2.bar(labels, tps_vals, color="#2e7d32", edgecolor="white")
    for bar, val in zip(bars2, tps_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.1f}", ha="center", va="bottom", fontsize=11)
    ax2.set_ylabel("Tokens per Second", fontsize=12)
    ax2.set_title("Generation Speed Comparison", fontsize=13, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()
    return fig
