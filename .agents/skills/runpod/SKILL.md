---
name: runpod
description: >-
  Manage RunPod cloud GPU infrastructure, query GPU availability and pricing,
  spin up/terminate pods, manage serverless endpoints, and run detached GPU benchmarks.
---

# RunPod Agent Skill

This skill equips the agent to manage RunPod GPU instances, inspect real-time pricing and availability, launch pods configured for vLLM and deep learning workloads, retrieve SSH connection strings, and terminate instances.

## Prerequisites
- **RunPod Account & API Key**: Generated from [RunPod Console Settings](https://console.runpod.io/user/settings).
- **Environment Variable**: `RUNPOD_API_KEY` set in the environment or `.env` file.
- **Python SDK**: `runpod` (`pip install runpod`).

## Common Workflows

### 1. Check GPU Availability & Pricing
Query available GPUs suitable for LLM serving (minimum 40GB VRAM recommended: L40S, A100 40GB/80GB):
```python
import os, runpod
runpod.api_key = os.environ.get("RUNPOD_API_KEY")

gpus = runpod.get_gpus()
for gpu in gpus:
    if "L40" in gpu["id"] or "A100" in gpu["id"]:
        print(f"GPU: {gpu['id']}, VRAM: {gpu['memoryInGb']}GB, Lowest Price: ${gpu.get('lowestPrice', {}).get('minimumBidPrice', 'N/A')}/hr")
```

### 2. Launch an LLM Serving Pod
Deploy an on-demand pod with the official RunPod PyTorch template:
- Template ID: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (or vLLM official image)
- Disk: 50GB container disk + 20GB volume disk
- GPU Type: `NVIDIA L40S` or `NVIDIA A100 80GB PCIe`

### 3. Connect via SSH & Run Experiments
Once the pod is `RUNNING`:
1. Retrieve the public IP and external SSH port from `runpod.get_pod(pod_id)`.
2. Connect via SSH:
   ```bash
   ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519
   ```
3. Run inside `tmux` so long benchmarks survive network disconnection:
   ```bash
   tmux new -s llm_serving
   git clone https://github.com/mridulpant2010/llm_serving.git
   cd llm_serving
   pip install -r requirements.txt
   python cli.py sweep --repetitions 1 --output-dir ./results_pilot
   ```

### 4. Terminate the Pod
Always stop/terminate pods when jobs complete to avoid idle billing:
```python
runpod.terminate_pod(pod_id)
```
