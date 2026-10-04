# Project Report: Balancing Memory and Compute in LLM Serving

## 1. Background

If we want to understand why serving Large Language Models (LLMs) efficiently is so difficult, it helps to look at the basic hardware context and constraints of a GPU. At the end of the day, an LLM is doing massive matrix multiplication. The speed of that process comes down to two physical limits: how fast the GPU can do math (Compute/FLOPS), and how fast it can move data from VRAM to the math units (Memory Bandwidth).

In production, generating text happens in two distinct phases:
1. **The Prefill Phase:** When a user sends a prompt, the GPU processes all the words at once. Because it's doing so much math simultaneously, the GPU's math units are pushed to their limit. This phase is **compute-bound**.
2. **The Decode Phase:** The model has to generate the response one word at a time. To predict a single word, the system has to load the entire model (often 16GB+ of weights) and the conversation history (the KV cache) from memory, do a tiny bit of math, and write the new word. This phase is severely **memory-bandwidth-bound**. 

**The Motivation:** This creates a major inefficiency. During the decode phase, the GPU's massive math units spend most of their time sitting idle, waiting for data to travel across the memory bus. To fix this, engineers usually try to do one of two things: 
*   Shrink the memory footprint (like compressing the KV cache) so they can batch more users together, forcing the GPU to do more math per data load. 
*   Give the idle math units a side-task, like guessing future tokens (Speculative Decoding) while they wait for data.

The motivation for this project comes from a simple realization: while these two tricks work great on their own, they actually fight against each other in real-world deployments. 

## 2. Problem Statement

This project addresses the inherent conflict between memory-saving optimizations and compute-utilization optimizations in LLM serving. 

**Scope:**
We are focusing on two specific bottlenecks and exploring how they interact:
*   *KV-Cache Precision (Topic 2):* Because decode throughput is limited by memory, compressing the KV cache down to 8-bit floating point (FP8) frees up VRAM and lets us handle more users at once. 
*   *Speculative Decoding Across Load (Topic 3):* Speculative decoding tries to use the GPU's spare compute to guess tokens, lowering response times. We are exploring three variants: Draft models, Self-speculation (EAGLE), and Draft-free (N-gram) methods.

**Key Challenges:**
The primary challenge is that speculative decoding relies entirely on having *idle compute* available. However, when we successfully compress the KV cache (Topic 2) to fit 50 or 100 concurrent users, the GPU isn't sitting idle anymore—the math units become fully saturated processing that massive batch of users. Therefore, maximizing our memory efficiency directly cannibalizes our compute efficiency. 

## 3. Expected Output

The intended outcome of this project is to produce a quantitative map of exactly how and when these two optimizations collide. Instead of just turning features "on," this project will achieve a precise understanding of system limits under load.

**What the project is expected to produce:**

1.  **The KV Cache Concurrency Limit:** Empirical data showing exactly how many extra users an FP8 cache buys us before the server crashes (OOM), and whether that extra concurrency actually translates to higher Total Throughput (Tokens/sec) or if it just hits a compute bottleneck. We will also produce MMLU and HumanEval scores to prove the 8-bit cache doesn't ruin the model's accuracy.
2.  **Speculative Decoding Performance Comparisons:** Hard data comparing Draft Models, EAGLE, and N-gram methods across a load sweep, measuring Time Per Output Token (TPOT), Token Acceptance Rate, and GPU Compute Utilization to see which method degrades the most gracefully under pressure.
3.  **The Crossover Point ($C^*$):** The ultimate deliverable is finding $C^*$—the exact number of concurrent users where speculative decoding stops helping and starts slowing the system down (where the Speedup Ratio drops below 1.0). 
4.  **A Production "Regime Map":** A final engineering guideline showing exactly how the $C^*$ crossover point shifts when we switch from an FP16 cache to an FP8 cache, giving practitioners a definitive rulebook for when to prioritize memory versus when to prioritize latency.

## 4. Existing Project Landscape

Right now, the LLM serving space is dominated by a few major frameworks and techniques:

*   **Serving Engines:** Tools like vLLM and SGLang are the current industry standards. vLLM popularized *PagedAttention*, which fixes memory fragmentation, but it doesn't natively compress the data. SGLang is great for optimizing complex workflows but still runs into the same fundamental memory limits.
*   **Memory Compression:** People have been using INT8 quantization for a while, but FP8 is quickly becoming the new standard for modern GPUs (like the Hopper architecture) because it compresses the cache without losing the dynamic range of the data. 
*   **Speculative Decoding Ecosystem:** There are three main ways people are doing this right now:
    1.  *Draft Models:* Using a tiny model to guess words for a massive model.
    2.  *Self-Speculation (EAGLE/Medusa):* Attaching extra "heads" to the main model so it can guess its own future words.
    3.  *Draft-Free (N-gram):* Using basic pattern matching to guess repeating phrases, which takes almost zero compute power.

**The Gap:** The problem with the current landscape is that most serving engines let you turn these features on, but they don't help you manage the conflict between them. There isn't much built-in logic to handle what happens when high user load makes speculation harmful.

## 5. Related Work

There's a lot of great foundational research here, but it tends to look at these problems in isolation.

For memory management, Kwon et al. (2023) built PagedAttention to stop memory waste, which was a huge leap forward. However, it's just allocating memory better, not compressing it. Newer papers on FP8 quantization show that compressing the bits is the logical next step, but they rarely test how this increased batch size impacts the GPU's overall compute availability.

For speculative decoding, Leviathan et al. (2022) proved the math behind making it lossless, and more recent work like Cai et al. (Medusa) and Li et al. (EAGLE) have made the guessing process incredibly efficient. 
*The catch:* Almost all of these academic papers test their speculative models with a Batch Size of 1. It looks great in a lab, but it doesn't reflect a real production server running continuous batching. By ignoring heavy load scenarios, they hide the point where compute contention makes their methods fail. Our project fills this gap by directly testing how memory gains from KV precision actively shift the breaking point of these speculative methods.

## 6. Justification for the Proposed Project

A GPU has a finite amount of memory and a finite amount of math it can do per second. Because of this, optimizing a server is basically a zero-sum game. 

From first principles, we know that if we compress the KV Cache, we free up memory. This lets us batch more users together. But if we batch more users together, we force the GPU to do a lot more math, eating up all the idle compute. Meanwhile, speculative decoding absolutely relies on having idle compute to make its guesses. 

This project is necessary because a lot of current engineering assumes that optimizations stack perfectly—if Trick A makes it 20% faster and Trick B makes it 20% faster, turning both on should be great. But physically, that's impossible at scale. If we successfully use KV compression to cram 100 users onto a GPU, it *has* to consume the idle compute that speculative decoding needs. They eventually destroy each other's benefits. 

We need to map out exactly where this collision happens. By finding the Crossover Point across different methods, this project will give engineers a practical "regime map." Instead of just turning on every optimization and hoping for the best, this map will show exactly when a server should prioritize cramming in more users, and when it should prioritize utilizing idle compute for lower latency.
