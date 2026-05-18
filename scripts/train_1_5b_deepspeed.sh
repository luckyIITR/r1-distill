#!/usr/bin/env bash
# Full fine-tuning of Qwen 2.5-1.5B with DeepSpeed ZeRO-2 on 2x L4.
set -euo pipefail

# ---------- Config ----------
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
DATASET="/root/data/sft_train_8k.jsonl"          # LOCAL, not /workspace
DS_CONFIG="configs/ds_zero2.json"                 # ZeRO-2, not ZeRO-3
OUTPUT_DIR="/root/checkpoints/qwen2.5-1.5b-r1-distill"  # LOCAL too
RUN_NAME="qwen2.5-1.5b-r1-distill-$(date +%Y%m%d-%H%M%S)"

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo ">>> Detected ${NUM_GPUS} GPUs"

if [[ "${NUM_GPUS}" -lt 2 ]]; then
  echo "!!! Need >=2 GPUs. Found ${NUM_GPUS}."
  exit 1
fi

# ---------- Attention ----------
if python -c "import flash_attn" 2>/dev/null; then
  ATTN_IMPL="flash_attn"
  echo ">>> Using flash-attn"
else
  ATTN_IMPL="sdpa"
  echo ">>> SDPA (flash-attn unavailable)"
fi

# ---------- Hyperparameters ----------
MAX_LENGTH=8192
LEARNING_RATE=2e-5
WARMUP_RATIO=0.05
EPOCHS=5
BATCH_PER_GPU=1
GRAD_ACCUM=$(( 8 / NUM_GPUS / BATCH_PER_GPU ))
WEIGHT_DECAY=0.0001

# ---------- Smoke ----------
if [[ "${SMOKE:-0}" == "1" ]]; then
  echo ">>> SMOKE MODE"
  EPOCHS=1
  MAX_STEPS_FLAG="--max_steps 30"
  REPORT_TO_FLAG="--report_to none"
  RUN_NAME="${RUN_NAME}-smoke"
  OUTPUT_DIR="${OUTPUT_DIR}-smoke"
else
  MAX_STEPS_FLAG=""
  REPORT_TO_FLAG="--report_to wandb"
fi

# ---------- Local cache for HF (CRITICAL) ----------
export HF_HOME=/root/.cache/huggingface
export TRANSFORMERS_CACHE=/root/.cache/huggingface
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets
mkdir -p "${HF_HOME}"

# ---------- Distributed env ----------
export NPROC_PER_NODE=${NUM_GPUS}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1

# NCCL diagnostics (keep until working, then remove)
export NCCL_DEBUG=WARN
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800
# Force P2P over PCIe (these L4s have no NVLink, but P2P helps)
export NCCL_P2P_DISABLE=0

# ---------- Launch ----------
echo ">>> Run: ${RUN_NAME}"
echo ">>> Effective batch size: $((BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM))"
echo ">>> Dataset: ${DATASET}"
echo ">>> Output: ${OUTPUT_DIR}"
echo ">>> DS config: ${DS_CONFIG}"

swift sft \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --attn_impl "${ATTN_IMPL}" \
  --deepspeed "${DS_CONFIG}" \
  --num_train_epochs ${EPOCHS} \
  --max_length ${MAX_LENGTH} \
  --per_device_train_batch_size ${BATCH_PER_GPU} \
  --gradient_accumulation_steps ${GRAD_ACCUM} \
  --learning_rate ${LEARNING_RATE} \
  --warmup_ratio ${WARMUP_RATIO} \
  --weight_decay ${WEIGHT_DECAY} \
  --lr_scheduler_type cosine \
  --gradient_checkpointing true \
  --save_strategy epoch \
  --save_total_limit 2 \
  --logging_steps 5 \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --seed 42 \
  --packing false \
  --dataset_num_proc 2 \
  --use_hf true \
  ${MAX_STEPS_FLAG} \
  ${REPORT_TO_FLAG}

echo ">>> Training complete. Checkpoint: ${OUTPUT_DIR}"
echo ">>> NOTE: Checkpoint is on local disk. Copy to /workspace before stopping pod:"
echo "    cp -r ${OUTPUT_DIR} /workspace/r1-distill/checkpoints/"