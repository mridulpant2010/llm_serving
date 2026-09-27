"""
LLM Serving Benchmark Harness (vLLM)

This script directly tests the two topics of the project report:
Topic 2: KV Cache Precision (comparing 'auto' vs 'fp8')
Topic 3: Speculative Decoding (comparing baseline vs draft model)

Requires: pip install vllm
"""

import time
import argparse
from typing import List, Optional

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("WARNING: vllm is not installed. Run `pip install vllm` on a Linux/Colab GPU instance.")
    LLM = None
    SamplingParams = None


def generate_dummy_prompts(batch_size: int, prompt_length: int = 128) -> List[str]:
    """Generate a batch of identical dummy prompts to stress the server."""
    # A simple repeating prompt to simulate a context
    base_text = "Explain the history of artificial intelligence and its impact on modern society. "
    # Roughly repeat to hit desired length (very simplified)
    repeats = max(1, prompt_length // 10)
    single_prompt = base_text * repeats
    
    return [single_prompt for _ in range(batch_size)]


def run_benchmark(
    target_model: str,
    kv_cache_dtype: str,
    speculative_model: Optional[str],
    batch_size: int,
    output_len: int = 128
):
    """
    Run a specific configuration and measure Throughput and Latency.
    """
    print(f"\n{'='*50}")
    print(f"Initializing vLLM Engine...")
    print(f"Target Model: {target_model}")
    print(f"KV Cache Precision: {kv_cache_dtype.upper()} (TOPIC 2)")
    print(f"Speculative Model: {speculative_model if speculative_model else 'None'} (TOPIC 3)")
    print(f"Batch Size (Concurrency): {batch_size}")
    print(f"{'='*50}")

    # Initialize the vLLM engine
    # This is where the magic happens: vLLM handles the PagedAttention automatically.
    # We pass the kv_cache_dtype to trigger Topic 2.
    # We pass speculative_model to trigger Topic 3.
    try:
        llm = LLM(
            model=target_model,
            kv_cache_dtype=kv_cache_dtype,
            speculative_model=speculative_model,
            num_speculative_tokens=5 if speculative_model else None,
            max_model_len=2048,
            gpu_memory_utilization=0.9, # Use 90% of GPU memory
            enforce_eager=True, # Often needed for smaller GPUs or quick testing
            trust_remote_code=True
        )
    except Exception as e:
        print(f"\n[!] Engine Initialization Failed (Likely OOM or unsupported hardware).")
        print(f"Error: {e}")
        return

    prompts = generate_dummy_prompts(batch_size)
    sampling_params = SamplingParams(
        temperature=0.0, # Greedy decoding for benchmarking
        max_tokens=output_len,
        ignore_eos=True  # Force it to generate exactly max_tokens
    )

    print(f"\nStarting benchmark with {batch_size} concurrent requests...")
    
    start_time = time.perf_counter()
    
    # Run the batch through vLLM
    try:
        outputs = llm.generate(prompts, sampling_params)
    except Exception as e:
        print(f"\n[!] Generation Failed (Likely OOM during decode).")
        print(f"Error: {e}")
        return
        
    end_time = time.perf_counter()
    
    # Calculate Metrics
    total_time = end_time - start_time
    total_tokens_generated = batch_size * output_len
    
    throughput = total_tokens_generated / total_time
    
    # TPOT (Time Per Output Token)
    # vLLM batches heavily, so TPOT is roughly total_time / output_len
    tpot_ms = (total_time / output_len) * 1000

    print(f"\n--- Benchmark Results ---")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Total Tokens Generated: {total_tokens_generated}")
    print(f"Throughput: {throughput:.2f} tokens/sec")
    print(f"Latency (TPOT): {tpot_ms:.2f} ms/token")
    
    # In vLLM, if speculative decoding was used, we can inspect acceptance rates
    # in the detailed metrics (though extracting it via Python API requires accessing engine stats)
    
    # Clean up memory so we can run the next test
    import torch
    import gc
    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Serving Benchmark")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--draft-model", type=str, default=None, help="Draft model for speculative decoding")
    parser.add_argument("--kv-dtype", type=str, choices=["auto", "fp8"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    
    args = parser.parse_args()
    
    if LLM is not None:
        run_benchmark(
            target_model=args.model,
            kv_cache_dtype=args.kv_dtype,
            speculative_model=args.draft_model,
            batch_size=args.batch_size
        )
