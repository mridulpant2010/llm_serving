#!/usr/bin/env bash
# One-time setup on a Linux GPU box (Vast.ai, RunPod, Lambda, IITH cluster).
# Usage:  ./scripts/setup_server.sh
set -e

echo "[1/5] GPU check"
nvidia-smi || { echo "nvidia-smi not found"; exit 1; }

echo "[2/5] Hugging Face authentication check"
if [ -n "$HF_TOKEN" ] && command -v huggingface-cli &> /dev/null; then
    echo "Logging into huggingface-cli with provided HF_TOKEN..."
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
else
    echo "Notice: HF_TOKEN not set; will use cached and public models."
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

# Models for the benchmark:
# Base: meta-llama/Meta-Llama-3-8B-Instruct (approved & already cached)
# Drafter: unsloth/Llama-3.2-1B-Instruct (public, identical weights & vocab, no 403)
# Eagle: yuhuili/EAGLE-LLaMA3-Instruct-8B (public Eagle head)
repos = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "unsloth/Llama-3.2-1B-Instruct",
    "yuhuili/EAGLE-LLaMA3-Instruct-8B",
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
        print(f"Notice: Failed to download {repo}: {e}")
        # If meta-llama 1B fails, try unsloth
        if "Llama-3.2-1B" in repo:
            alt = "unsloth/Llama-3.2-1B-Instruct"
            print(f"Retrying with public mirror {alt} ...")
            snapshot_download(alt, token=token, allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "tokenizer*"])
            print(f"Successfully cached {alt}.")
        else:
            raise
EOF

python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0), '| sm', torch.cuda.get_device_capability(0))"
echo ""
echo "=========================================================="
echo " Setup complete! Ready to start benchmark.                "
echo " Next command: tmux new -s eval ; ./scripts/run_experiments.sh budget "
echo "=========================================================="
