#!/usr/bin/env bash
set -euo pipefail

TRAIN_FILE="train_dllm.jsonl"
MODEL_PATH="qwen3-0_6b"
OUTPUT_DIR="OUTPUT"
mkdir -p "${OUTPUT_DIR}"

# GPUs (or set CUDA_VISIBLE_DEVICES beforehand)
: "${NPROC:=$( (nvidia-smi -L | wc -l) 2>/dev/null || echo 1 )}"
: "${MASTER_PORT:=29500}"

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1


ARGS=(
  --train_file "${TRAIN_FILE}"
  --model_name_or_path "${MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --max_seq_length 12000
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 8
  --learning_rate 2e-5
  --num_train_epochs 3
  --warmup_ratio 0.03
  --logging_steps 10
  --save_steps 1000
  --save_total_limit 3
  --lr_scheduler_type cosine
  --bf16
  --gradient_checkpointing
)

torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  dllm_train_lora.py "${ARGS[@]}"
