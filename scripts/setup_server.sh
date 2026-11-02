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
    echo "Notice: HF_TOKEN not set or huggingface-cli absent; using ungated public model weights."
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

# Model targets: uses public ungated mirrors to guarantee zero 403 authorization blockage
repos = [
    ("NousResearch/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("unsloth/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"),
    ("yuhuili/EAGLE-LLaMA3-Instruct-8B", "yuhuili/EAGLE-LLaMA3-Instruct-8B"),
]

for mirror, official in repos:
    # Try official first if token is provided, else fall back to ungated mirror
    downloaded = False
    candidates = [official, mirror] if token else [mirror]
    for repo in candidates:
        print(f"\n---> Downloading {repo} ...")
        try:
            snapshot_download(
                repo,
                token=token,
                allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "tokenizer*"]
            )
            print(f"Successfully cached {repo}.")
            downloaded = True
            break
        except Exception as e:
            print(f"Notice: Failed downloading {repo}: {e}")

    if not downloaded:
        print(f"ERROR: Could not download either {mirror} or {official}!", file=sys.stderr)
        sys.exit(1)
EOF

python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0), '| sm', torch.cuda.get_device_capability(0))"
echo "Setup done. Next: tmux new -s eval ; ./scripts/run_experiments.sh pilot"
