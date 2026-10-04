# Project Report: Balancing Memory and Compute in LLM Serving

## 1. Background

- With chatbots, coding assistants, and other LLM-powered applications going mainstream, the demand for hosting these models in the cloud has grown rapidly.
- Running inference on large language models is expensive — it requires high-end GPUs, and processing each request costs real money.
- This economic pressure drives the need to serve more users per GPU. The more requests a single GPU handles concurrently, the lower the cost per request.

- An LLM is essentially performing matrix multiplications, and how fast it can do that depends on two hardware limits:
  - **Compute throughput** — how many math operations the GPU can do per second (FLOPS).
  - **Memory bandwidth** — how fast data moves from VRAM to the processing cores (Bytes/sec).

- Text generation splits into two phases:
  - **Prefill phase:** The model processes the entire input prompt in one shot. All tokens are handled in parallel, keeping the GPU's math units busy. This part is compute-bound.
  - **Decode phase:** The model generates output tokens one at a time. Each token requires loading the full model weights and the accumulated KV cache from memory, performing a small amount of computation, and writing out the result. Generating $K$ tokens means $K$ serial passes through the model. This part is memory-bandwidth-bound.

- The **KV cache** makes the memory problem worse:
  - As the model generates tokens, it stores key and value tensors from the attention mechanism so it doesn't have to recompute them.
  - These tensors grow with every token and every concurrent request.
  - In a typical setup, the KV cache can consume roughly a third of the GPU's memory.
  - Since model weights are static, it's really the KV cache that determines how many requests can run at the same time.

- During decode, the GPU's compute units are mostly idle — they're waiting on memory reads. Engineers have two broad strategies to deal with this:
  - **Compress the KV cache** (e.g., store it in 8-bit instead of 16-bit) to free up memory, batch more requests together, and raise throughput.
  - **Use speculative decoding** to put the idle compute units to work by guessing future tokens in advance, reducing latency for individual requests.

- Both approaches work well independently. The problem — and the motivation behind this project — is that they conflict with each other once you try to use them together under real production load.

## 2. Problem Statement

- This project investigates how two common LLM serving optimizations interact under load, and at what point they start working against each other.

- **The memory side:**
  - Serving many concurrent users requires fitting all their KV caches into GPU memory.
  - The numbers get large quickly — a 13B-parameter model needs roughly 1.6 GB of KV cache per request, which eats up close to 30% of a 40 GB GPU for just one user.
  - Sequence lengths are unpredictable — prompts vary in size and outputs grow token by token, so memory needs can't be known in advance.
  - Older serving systems pre-allocated memory based on worst-case sequence lengths, wasting space through internal and external fragmentation.
  - Modern engines like vLLM have mostly fixed the fragmentation problem, but even with perfect allocation, the uncompressed cache is simply too large to fit many concurrent requests.

- **The compute side:**
  - Because decode generates one token at a time, each step involves loading a huge amount of data for very little arithmetic work.
  - The GPU's compute units are heavily underutilized.
  - Speculative decoding tries to exploit this gap — it uses spare compute to draft multiple candidate tokens ahead of time, then verifies them in a single pass.

- **Scope — what we are focusing on:**
  - *KV-Cache Precision:* Storing the KV cache in FP8 instead of FP16 to roughly halve the cache footprint and fit more concurrent requests. The question is whether that extra capacity actually translates to better throughput, or whether something else becomes the bottleneck.
  - *Speculative Decoding Across Load:* Testing three speculation approaches — a separate draft model, self-speculation with prediction heads (EAGLE), and simple n-gram matching — to find where each method stops helping as server load increases.

- **The core challenge:**
  - Compressing the KV cache lets us pack more users onto the GPU, which means the compute units get busier handling a larger batch.
  - But speculative decoding needs those compute units to be idle so it can run its guessing process.
  - So the better we do on the memory side, the less room there is for speculation to work.
  - We want to find out exactly where this trade-off tips over.

## 3. Expected Output

- The goal is to produce concrete, measurable results that show how these two optimizations behave — individually and together — as load changes.
- We are not just trying to show that they work; we want to map out the boundaries of when they work and when they don't.

- **KV cache concurrency data:**
  - Note: this is specifically about compressing the *cache*, not the model weights. Weight quantization is a separate lever — here the variable is the precision of the stored key-value tensors during inference.
  - Measure how many concurrent requests the server can handle with FP16 versus FP8 KV caches before running out of memory.
  - Check whether the extra headroom from FP8 actually results in higher throughput (tokens per second across all users), or whether the system hits a compute wall instead.
  - Sweep across different context lengths (e.g., 512, 2048, 4096, 8192) to see how the benefit of FP8 changes — shorter contexts may not benefit much since the cache is small anyway, while longer contexts should see a bigger gain.
  - Run the model through MMLU and HumanEval to check for quality degradation, and specifically identify *where* it becomes visible — whether it shows up at longer context lengths, on reasoning-heavy tasks, or only past a certain cache size.

- **Comparison of speculative decoding methods under load:**
  - Benchmark draft-model speculation, EAGLE-style self-speculation, and n-gram matching side by side, sweeping concurrency from low to high.
  - Track time per output token (TPOT), token acceptance rate, and GPU compute utilization for each method.
  - Measure the memory overhead of each speculation method — a draft model sits in VRAM alongside the target model, directly eating into the space available for KV cache. EAGLE adds extra parameters but much less than a full draft model. N-gram matching costs essentially nothing. We want to quantify how each method's memory footprint reduces the max concurrency we established in Topic 2, and whether the latency gains are worth that trade-off.
  - Identify which approach holds up best as the server gets busier.

- **The crossover point ($C^*$):**
  - Find the concurrency level where speculative decoding flips from being helpful to harmful.
  - This is the point where the speedup ratio drops below 1.0 and speculation actually slows things down compared to the baseline.

- **A regime map:**
  - Show how $C^*$ shifts when switching from an FP16 cache to an FP8 cache.
  - If FP8 lets you run at higher concurrency, and higher concurrency kills speculation, then the crossover point should move.
  - Documenting this relationship gives engineers a practical guide for deciding when to prioritize memory savings and when to prioritize latency reduction.
