#!/usr/bin/env bash
# Full fine-tuning of Qwen 2.5-7B with DeepSpeed ZeRO-3.
# Designed for 4x or 8x A100-80GB.
#
# Usage:
#   bash scripts/train_7b.sh                    # full run
#   SMOKE=1 bash scripts/train_7b.sh            # smoke test (50 steps, 2 epochs)
#
# Required env:
#   WANDB_API_KEY, HF_TOKEN, WANDB_PROJECT
set -euo pipefail

# ---------- Config ----------
MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASET="data/processed/sft_train.jsonl"
DS_CONFIG="configs/ds_zero3.json"
OUTPUT_DIR="checkpoints/qwen2.5-7b-r1-distill"
RUN_NAME="qwen2.5-7b-r1-distill-$(date +%Y%m%d-%H%M%S)"

# Auto-detect GPU count
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo ">>> Detected ${NUM_GPUS} GPUs"

# ---------- Hyperparameters ----------
MAX_LENGTH=20480
LEARNING_RATE=1e-5
WARMUP_RATIO=0.05
EPOCHS=5
BATCH_PER_GPU=1
GRAD_ACCUM=$(( 16 / NUM_GPUS ))  # effective batch ~16
WEIGHT_DECAY=0.0001

# ---------- Smoke test overrides ----------
if [[ "${SMOKE:-0}" == "1" ]]; then
  echo ">>> SMOKE MODE: 2 epochs, max 50 steps, no W&B"
  EPOCHS=2
  MAX_STEPS_FLAG="--max_steps 50"
  REPORT_TO="--report_to none"
  RUN_NAME="${RUN_NAME}-smoke"
  OUTPUT_DIR="${OUTPUT_DIR}-smoke"
else
  MAX_STEPS_FLAG=""
  REPORT_TO="--report_to wandb"
fi

# ---------- Launch ----------
echo ">>> Starting training: ${RUN_NAME}"
echo ">>> Effective batch size: $((BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM))"
echo ">>> Output: ${OUTPUT_DIR}"

export NPROC_PER_NODE=${NUM_GPUS}
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
# Speed up HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

swift sft \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --train_type full \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
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
  --dataset_num_proc 4 \
  ${MAX_STEPS_FLAG} \
  ${REPORT_TO}

echo ">>> Training complete. Checkpoint at: ${OUTPUT_DIR}"