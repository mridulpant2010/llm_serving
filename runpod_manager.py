"""
RunPod Automation Manager for LLM Serving Experiments
Enables querying GPU pricing, launching pods with PyTorch/vLLM, inspecting SSH info, and stopping pods.
"""

import os
import sys
import argparse
import json

try:
    import runpod
except ImportError:
    runpod = None

def get_api_key():
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key and os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("RUNPOD_API_KEY="):
                    api_key = line.strip().split("=", 1)[1].strip("'\"")
    return api_key

def check_sdk():
    if runpod is None:
        print("[ERROR] 'runpod' package is not installed.")
        print("Please run: pip install runpod")
        sys.exit(1)
    
    key = get_api_key()
    if not key:
        print("[ERROR] RUNPOD_API_KEY is not set.")
        print("Set it via environment variable: set RUNPOD_API_KEY=your_key")
        print("Or add RUNPOD_API_KEY=your_key to a .env file in this directory.")
        print("Get your API key at: https://console.runpod.io/user/settings")
        sys.exit(1)
    
    runpod.api_key = key

def list_gpus(min_vram=24):
    check_sdk()
    print(f"\n[RunPod] Querying available GPUs with >= {min_vram}GB VRAM...\n")
    try:
        gpus = runpod.get_gpus()
        print(f"{'GPU Type':<30} | {'VRAM (GB)':<10} | {'Lowest Price/hr':<16} | {'Max Instance Price':<18}")
        print("-" * 82)
        for g in gpus:
            vram = g.get("memoryInGb", 0)
            if vram >= min_vram:
                lowest = g.get("lowestPrice", {})
                price = lowest.get("minimumBidPrice") or lowest.get("uninterruptablePrice") or "N/A"
                if isinstance(price, (int, float)):
                    price_str = f"${price:.2f}"
                else:
                    price_str = str(price)
                
                max_price = g.get("maxPrice") or "N/A"
                if isinstance(max_price, (int, float)):
                    max_price_str = f"${max_price:.2f}"
                else:
                    max_price_str = str(max_price)

                print(f"{g.get('id', 'Unknown'):<30} | {vram:<10} | {price_str:<16} | {max_price_str:<18}")
        print("\nRecommended for Llama-3-8B + FP8 + Speculative Decoding: 'NVIDIA L40S' or 'NVIDIA A100 80GB PCIe'")
    except Exception as e:
        print(f"[ERROR] Failed to query GPUs: {e}")

def list_pods():
    check_sdk()
    print("\n[RunPod] Checking active pods...\n")
    try:
        pods = runpod.get_pods()
        if not pods:
            print("No active pods found.")
            return
        
        print(f"{'Pod ID':<20} | {'Name':<25} | {'GPU':<20} | {'Status':<12}")
        print("-" * 82)
        for p in pods:
            gpu_name = p.get("machine", {}).get("gpuDisplayName", "Unknown") if isinstance(p.get("machine"), dict) else "Unknown"
            print(f"{p.get('id', ''):<20} | {p.get('name', '')[:24]:<25} | {gpu_name:<20} | {p.get('desiredStatus', ''):<12}")
    except Exception as e:
        print(f"[ERROR] Failed to list pods: {e}")

def launch_pod(gpu_type="NVIDIA L40S", name="llm-serving-eval"):
    check_sdk()
    print(f"\n[RunPod] Launching Pod '{name}' with GPU '{gpu_type}'...")
    
    # Official standard PyTorch 2.4 + CUDA 12.4 template
    image_name = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    
    try:
        pod = runpod.create_pod(
            name=name,
            image_name=image_name,
            gpu_type_id=gpu_type,
            cloud_type="SECURE",  # or "COMMUNITY" for cheaper spot rates
            container_disk_in_gb=50,
            volume_in_gb=30,
            ports="22/tcp,8000/http",
            start_ssh=True
        )
        print("\n Pod launched successfully!")
        print(f"Pod ID: {pod.get('id')}")
        print("Waiting 15-30 seconds for container initialization...")
        print(f"To monitor pod status: python runpod_manager.py status")
        print(f"To terminate when done: python runpod_manager.py terminate {pod.get('id')}")
    except Exception as e:
        print(f"[ERROR] Failed to launch pod: {e}")

def terminate_pod(pod_id):
    check_sdk()
    print(f"\n[RunPod] Terminating Pod '{pod_id}'...")
    try:
        res = runpod.terminate_pod(pod_id)
        print(f"Pod {pod_id} terminated. Response: {res}")
    except Exception as e:
        print(f"[ERROR] Failed to terminate pod: {e}")

def main():
    parser = argparse.ArgumentParser(description="RunPod Infrastructure Manager for LLM Serving")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    subparsers.add_parser("gpus", help="List available GPUs and pricing")
    subparsers.add_parser("status", help="List active pods and status")
    
    launch_parser = subparsers.add_parser("launch", help="Launch a GPU pod")
    launch_parser.add_argument("--gpu", default="NVIDIA L40S", help="GPU type (e.g. 'NVIDIA L40S', 'NVIDIA A100 80GB PCIe')")
    launch_parser.add_argument("--name", default="llm-serving-eval", help="Pod name")

    term_parser = subparsers.add_parser("terminate", help="Terminate a pod")
    term_parser.add_argument("pod_id", help="Pod ID to terminate")

    args = parser.parse_args()

    if args.command == "gpus":
        list_gpus()
    elif args.command == "status":
        list_pods()
    elif args.command == "launch":
        launch_pod(gpu_type=args.gpu, name=args.name)
    elif args.command == "terminate":
        terminate_pod(args.pod_id)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
