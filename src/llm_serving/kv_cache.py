"""
KV Cache Analysis Module
========================

Provides utilities for inspecting, measuring, and comparing KV cache behaviour
across different precision formats (FP32, FP16, BF16, INT8, simulated FP8).

Designed to run in a Jupyter notebook on a laptop with a small model (e.g. GPT-2)
so you can validate the full pipeline before moving to a server with a large model.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Data classes for clean result tracking
# ---------------------------------------------------------------------------

@dataclass
class CacheProfile:
    """Snapshot of a KV cache's shape and memory usage."""
    num_layers: int
    num_heads: int
    head_dim: int
    seq_length: int
    dtype: str
    total_bytes: int
    size_mb: float


@dataclass
class GenerationResult:
    """Captures latency and throughput from a single generation run."""
    prompt: str
    num_tokens_generated: int
    total_time_sec: float
    time_to_first_token_sec: float
    tokens_per_sec: float
    time_per_output_token_ms: float
    cache_profile: Optional[CacheProfile] = None


@dataclass
class PrecisionComparison:
    """Side-by-side comparison of two cache precisions."""
    original_dtype: str
    compressed_dtype: str
    original_size_mb: float
    compressed_size_mb: float
    memory_saved_pct: float
    max_seq_at_original: int
    max_seq_at_compressed: int


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str = "gpt2", device: str = "cpu"):
    """
    Load a causal LM and its tokenizer.

    For local notebook work, use "gpt2" (124M params, runs on CPU).
    On the server, swap to "meta-llama/Meta-Llama-3-8B-Instruct" and
    set device="cuda".
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    model = model.to(device)
    model.eval()

    # GPT-2 doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loaded {model_id} on {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"  Weight dtype: {next(model.parameters()).dtype}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# KV Cache inspection
# ---------------------------------------------------------------------------

def extract_kv_cache(model, tokenizer, prompt: str, device: str = "cpu"):
    """
    Run a single forward pass and return the raw KV cache tensors.

    Returns:
        kv_cache: tuple of (key, value) tensors per layer
        outputs: the full model output object
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    kv_cache = outputs.past_key_values
    return kv_cache, outputs


def profile_kv_cache(kv_cache) -> CacheProfile:
    """
    Measure the shape and memory footprint of a KV cache.

    The KV cache is a tuple of (key_tensor, value_tensor) per layer.
    Key shape is typically: [batch_size, num_heads, seq_length, head_dim]
    """
    total_bytes = 0
    for layer in kv_cache:
        for tensor in layer:  # key and value
            total_bytes += tensor.numel() * tensor.element_size()

    # Pull shape info from the first layer's key tensor
    first_key = kv_cache[0][0]
    _, num_heads, seq_length, head_dim = first_key.shape

    return CacheProfile(
        num_layers=len(kv_cache),
        num_heads=num_heads,
        head_dim=head_dim,
        seq_length=seq_length,
        dtype=str(first_key.dtype),
        total_bytes=total_bytes,
        size_mb=total_bytes / (1024 * 1024),
    )


def print_cache_profile(profile: CacheProfile):
    """Pretty-print a cache profile."""
    print(f"  Layers: {profile.num_layers}")
    print(f"  Heads per layer: {profile.num_heads}")
    print(f"  Head dimension: {profile.head_dim}")
    print(f"  Sequence length: {profile.seq_length}")
    print(f"  Dtype: {profile.dtype}")
    print(f"  Total size: {profile.size_mb:.4f} MB ({profile.total_bytes:,} bytes)")


# ---------------------------------------------------------------------------
# KV Cache precision conversion (simulate quantization)
# ---------------------------------------------------------------------------

def convert_kv_cache_precision(kv_cache, target_dtype: torch.dtype):
    """
    Convert the entire KV cache to a different precision.

    This simulates what happens when you set --kv-cache-dtype fp8 or int8
    on a real serving engine. The conversion is lossy for int8 (we quantize
    per-tensor with simple rounding), which lets you measure quality impact.

    Supported target dtypes:
        torch.float16   — FP16 (2 bytes)
        torch.bfloat16  — BF16 (2 bytes)
        torch.int8      — INT8 (1 byte, simulated quantization)
        torch.float32   — FP32 (4 bytes, for reference)
    """
    converted = []
    for layer in kv_cache:
        key, value = layer

        if target_dtype == torch.int8:
            # Simple per-tensor symmetric quantization
            k_scale = key.abs().max() / 127.0
            v_scale = value.abs().max() / 127.0
            key_q = (key / k_scale).round().clamp(-128, 127).to(torch.int8)
            val_q = (value / v_scale).round().clamp(-128, 127).to(torch.int8)
            converted.append((key_q, val_q, k_scale, v_scale))
        else:
            converted.append((key.to(target_dtype), value.to(target_dtype)))

    return converted


def dequantize_int8_cache(quantized_cache):
    """
    Convert an INT8-quantized KV cache back to FP32 for quality comparison.

    Each layer is stored as (key_int8, value_int8, key_scale, value_scale).
    """
    restored = []
    for layer in quantized_cache:
        key_q, val_q, k_scale, v_scale = layer
        key_restored = key_q.to(torch.float32) * k_scale
        val_restored = val_q.to(torch.float32) * v_scale
        restored.append((key_restored, val_restored))
    return restored


def compare_precisions(kv_cache, target_dtype: torch.dtype) -> PrecisionComparison:
    """
    Compare the original KV cache against a compressed version.

    Returns a PrecisionComparison with memory savings and estimated
    max sequence lengths (assuming a fixed memory budget).
    """
    original_profile = profile_kv_cache(kv_cache)

    compressed = convert_kv_cache_precision(kv_cache, target_dtype)

    # Calculate compressed size
    compressed_bytes = 0
    for layer in compressed:
        for item in layer:
            if isinstance(item, torch.Tensor):
                compressed_bytes += item.numel() * item.element_size()
            # scales are scalar floats, negligible

    compressed_mb = compressed_bytes / (1024 * 1024)
    saved_pct = (1 - compressed_mb / original_profile.size_mb) * 100

    # Estimate: if we have 10 GB for KV cache, how many tokens fit?
    budget_bytes = 10 * 1024 * 1024 * 1024  # 10 GB
    bytes_per_token_original = original_profile.total_bytes / original_profile.seq_length
    bytes_per_token_compressed = compressed_bytes / original_profile.seq_length

    max_seq_original = int(budget_bytes / bytes_per_token_original) if bytes_per_token_original > 0 else 0
    max_seq_compressed = int(budget_bytes / bytes_per_token_compressed) if bytes_per_token_compressed > 0 else 0

    return PrecisionComparison(
        original_dtype=original_profile.dtype,
        compressed_dtype=str(target_dtype),
        original_size_mb=original_profile.size_mb,
        compressed_size_mb=compressed_mb,
        memory_saved_pct=saved_pct,
        max_seq_at_original=max_seq_original,
        max_seq_at_compressed=max_seq_compressed,
    )


# ---------------------------------------------------------------------------
# Quantization error measurement (quality proxy)
# ---------------------------------------------------------------------------

def measure_quantization_error(kv_cache, target_dtype: torch.dtype):
    """
    Measure the numerical error introduced by quantizing the KV cache.

    Returns per-layer mean squared error and max absolute error,
    which serves as a proxy for output quality degradation.
    """
    compressed = convert_kv_cache_precision(kv_cache, target_dtype)

    if target_dtype == torch.int8:
        restored = dequantize_int8_cache(compressed)
    else:
        restored = [(k.to(torch.float32), v.to(torch.float32)) for k, v in compressed]

    results = []
    for i, (orig_layer, rest_layer) in enumerate(zip(kv_cache, restored)):
        orig_key, orig_val = orig_layer
        rest_key, rest_val = rest_layer

        key_mse = ((orig_key.float() - rest_key.float()) ** 2).mean().item()
        val_mse = ((orig_val.float() - rest_val.float()) ** 2).mean().item()
        key_max_err = (orig_key.float() - rest_key.float()).abs().max().item()
        val_max_err = (orig_val.float() - rest_val.float()).abs().max().item()

        results.append({
            "layer": i,
            "key_mse": key_mse,
            "value_mse": val_mse,
            "key_max_abs_error": key_max_err,
            "value_max_abs_error": val_max_err,
        })

    return results


# ---------------------------------------------------------------------------
# Latency measurement (TPOT / TTFT)
# ---------------------------------------------------------------------------

def measure_generation_latency(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    device: str = "cpu",
) -> GenerationResult:
    """
    Generate tokens and measure Time-To-First-Token (TTFT) and
    Time-Per-Output-Token (TPOT).

    This is the core latency metric for your project.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Warm-up run (important for GPU timing)
    if device != "cpu":
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1)
        torch.cuda.synchronize()

    # Timed generation
    if device != "cpu":
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
            use_cache=True,
        )

    if device != "cpu":
        torch.cuda.synchronize()

    t_end = time.perf_counter()

    num_generated = output_ids.shape[1] - input_len
    total_time = t_end - t_start

    # Measure TTFT separately (generate just 1 token)
    if device != "cpu":
        torch.cuda.synchronize()

    t_ttft_start = time.perf_counter()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=True)

    if device != "cpu":
        torch.cuda.synchronize()

    t_ttft_end = time.perf_counter()
    ttft = t_ttft_end - t_ttft_start

    tpot_ms = (total_time / num_generated * 1000) if num_generated > 0 else 0
    tps = num_generated / total_time if total_time > 0 else 0

    return GenerationResult(
        prompt=prompt,
        num_tokens_generated=num_generated,
        total_time_sec=total_time,
        time_to_first_token_sec=ttft,
        tokens_per_sec=tps,
        time_per_output_token_ms=tpot_ms,
    )


# ---------------------------------------------------------------------------
# Batch simulation (concurrency)
# ---------------------------------------------------------------------------

def simulate_batch_concurrency(
    model,
    tokenizer,
    prompt: str,
    batch_sizes: list[int],
    max_new_tokens: int = 30,
    device: str = "cpu",
) -> list[dict]:
    """
    Simulate different concurrency levels by creating batches of the
    same prompt and measuring throughput and memory at each level.

    This finds the OOM point — the max batch size the hardware can handle.
    """
    results = []

    for bs in batch_sizes:
        # Create a batch by repeating the prompt
        inputs = tokenizer(
            [prompt] * bs,
            return_tensors="pt",
            padding=True,
        ).to(device)

        try:
            if device != "cpu":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            t_start = time.perf_counter()

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )

            if device != "cpu":
                torch.cuda.synchronize()

            t_end = time.perf_counter()

            total_time = t_end - t_start
            input_len = inputs["input_ids"].shape[1]
            tokens_per_request = output_ids.shape[1] - input_len
            total_tokens = tokens_per_request * bs

            peak_mem_mb = None
            if device != "cpu":
                peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            results.append({
                "batch_size": bs,
                "total_tokens": total_tokens,
                "total_time_sec": total_time,
                "throughput_tps": total_tokens / total_time,
                "latency_per_token_ms": (total_time / tokens_per_request * 1000)
                    if tokens_per_request > 0 else 0,
                "peak_memory_mb": peak_mem_mb,
                "status": "ok",
            })

            print(f"  Batch {bs}: {total_tokens / total_time:.1f} tok/s, "
                  f"{total_time:.2f}s total"
                  + (f", {peak_mem_mb:.0f} MB peak" if peak_mem_mb else ""))

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                results.append({
                    "batch_size": bs,
                    "status": "OOM",
                    "error": str(e),
                })
                print(f"  Batch {bs}: OOM! This is your concurrency ceiling.")
                if device != "cpu":
                    torch.cuda.empty_cache()
                break
            else:
                raise

    return results


# ---------------------------------------------------------------------------
# Context length sweep
# ---------------------------------------------------------------------------

def sweep_context_lengths(
    model,
    tokenizer,
    context_lengths: list[int],
    device: str = "cpu",
) -> list[dict]:
    """
    Measure KV cache size and generation latency across different
    context lengths.

    This answers the Topic 2 question: "How does the benefit change
    with context length?"
    """
    results = []

    for ctx_len in context_lengths:
        # Build a prompt of approximately the target length
        base = "The quick brown fox jumps over the lazy dog. "
        tokens_per_repeat = len(tokenizer.encode(base, add_special_tokens=False))
        repeats = max(1, ctx_len // tokens_per_repeat)
        prompt = base * repeats

        # Truncate to exact length
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)[:ctx_len]
        prompt = tokenizer.decode(token_ids)

        actual_len = len(tokenizer.encode(prompt))

        # Extract cache and measure
        kv_cache, _ = extract_kv_cache(model, tokenizer, prompt, device)
        profile = profile_kv_cache(kv_cache)

        # Also get INT8 comparison
        comparison = compare_precisions(kv_cache, torch.int8)

        results.append({
            "target_ctx_len": ctx_len,
            "actual_ctx_len": actual_len,
            "cache_size_fp32_mb": profile.size_mb,
            "cache_size_int8_mb": comparison.compressed_size_mb,
            "memory_saved_pct": comparison.memory_saved_pct,
            "bytes_per_token_fp32": profile.total_bytes / actual_len,
            "bytes_per_token_int8": (comparison.compressed_size_mb * 1024 * 1024) / actual_len,
        })

        print(f"  ctx={actual_len:>5d}: "
              f"FP32={profile.size_mb:.3f} MB, "
              f"INT8={comparison.compressed_size_mb:.3f} MB, "
              f"saved={comparison.memory_saved_pct:.1f}%")

        # Clean up
        del kv_cache
        if device != "cpu":
            torch.cuda.empty_cache()

    return results
