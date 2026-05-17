#!/usr/bin/env bash
# Run this ONCE on a fresh Runpod instance.
# Usage: bash scripts/runpod_setup.sh
set -euo pipefail

# Where to install
WORKDIR="${WORKDIR:-/workspace/r1-distill}"

# Clone if not present
if [[ ! -d "${WORKDIR}" ]]; then
  cd /workspace
  git clone "${REPO_URL:-https://github.com/luckyIITR/r1-distill.git}" r1-distill
fi
cd "${WORKDIR}"

# Verify GPUs visible
echo ">>> GPU check:"
nvidia-smi --list-gpus

# Install Python deps
echo ">>> Installing Python dependencies (this takes 5-10 min)"
pip install -U pip wheel
pip install -e ".[dev]"
pip install hf_transfer flash-attn==2.6.3 --no-build-isolation

# Verify ms-swift
echo ">>> ms-swift version:"
python -c "import swift; print(swift.__version__)"

# Set up env vars (you must export HF_TOKEN, WANDB_API_KEY before running this)
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "!!! HF_TOKEN not set. Run: export HF_TOKEN=hf_..."
  exit 1
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "!!! WANDB_API_KEY not set."
  exit 1
fi
huggingface-cli login --token "${HF_TOKEN}"
wandb login "${WANDB_API_KEY}"

# Verify training data present
if [[ ! -f "data/processed/sft_train.jsonl" ]]; then
  echo "!!! data/processed/sft_train.jsonl missing — upload it before training"
  exit 1
fi
echo ">>> Found $(wc -l < data/processed/sft_train.jsonl) training examples"

# Pre-download the model to avoid timing out at training start
echo ">>> Pre-downloading Qwen2.5-7B-Instruct (~15GB)..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir_use_symlinks=False)
print('Model downloaded.')
"

echo ">>> Setup complete. Run smoke test next:"
echo "    SMOKE=1 bash scripts/train_7b.sh"