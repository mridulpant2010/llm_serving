#!/usr/bin/env bash
# One-time setup on a Linux GPU box (Vast.ai, RunPod, Lambda, IITH cluster).
# Usage:  export HF_TOKEN=hf_xxx ; ./scripts/setup_server.sh
set -e

echo "[1/5] GPU check"
nvidia-smi || { echo "nvidia-smi not found"; exit 1; }

echo "[2/5] Hugging Face token check (Llama-3 is gated)"
if [ -z "$HF_TOKEN" ]; then
    echo "HF_TOKEN is not set. Run: export HF_TOKEN=hf_xxx  and accept the licence at"
    echo "  https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"
    echo "  https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct"
    exit 1
fi

# Authenticate huggingface-cli so vllm and all tools have access
if command -v huggingface-cli &> /dev/null; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
fi

echo "[3/5] Python packages"
pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps

echo "[4/5] WikiText-2 corpus"
python scripts/prepare_corpus.py

echo "[5/5] Pre-downloading model weights (so download time is not counted as a server-start failure)"
python - <<'EOF'
import os
import sys
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN")
repos = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "yuhuili/EAGLE-LLaMA3-Instruct-8B"
]

for repo in repos:
    print(f"\n---> Downloading {repo} ...")
    try:
        snapshot_download(
            repo,
            token=token,
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "tokenizer*"]
        )
        print(f"Successfully cached {repo}.")
    except Exception as e:
        print(f"\n[ERROR] Failed to download {repo}: {e}", file=sys.stderr)
        if "403" in str(e) or "GatedRepoError" in str(e):
            print("\n" + "="*70, file=sys.stderr)
            print("ACCESS PERMISSION REQUIRED FOR THIS HUGGING FACE MODEL:", file=sys.stderr)
            print(f"1. Make sure you are logged in to Hugging Face with the account for this token.", file=sys.stderr)
            print(f"2. Visit: https://huggingface.co/{repo}", file=sys.stderr)
            print(f"3. Submit the access request form (instant approval).", file=sys.stderr)
            print(f"4. Ensure your token has 'Read' permission at: https://huggingface.co/settings/tokens", file=sys.stderr)
            print("="*70 + "\n", file=sys.stderr)
        sys.exit(1)
EOF

python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0), '| sm', torch.cuda.get_device_capability(0))"
echo "Setup done. Next: tmux new -s eval ; ./scripts/run_experiments.sh pilot"
