FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace/llm_serving

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tmux \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository code
COPY . .
RUN pip install --no-cache-dir -e .

# Prepare benchmark corpus
RUN python scripts/prepare_corpus.py

CMD ["bash"]
