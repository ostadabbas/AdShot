#!/bin/bash
#============================================================================
# LoRA SFT of Qwen3-VL-8B on shot-selection, LoRA rank r=32 (alpha=2r=64).
# Per-epoch eval on data/eval.csv; final test eval on data/test_final.csv.
# gpu partition caps at 8h; a full 3-epoch run is ~16h -> resubmit the SAME
# script to auto-resume from the latest checkpoint (--resume auto default).
#============================================================================
#SBATCH --job-name=sft_lora_r32
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --mem=96GB
#SBATCH --output=logs/sft_lora_r32_%j.out
#SBATCH --error=logs/sft_lora_r32_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rastogi.o@northeastern.edu

set -euo pipefail

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-interactive}  Node: ${SLURM_NODELIST:-$(hostname)}  r=32"
echo "Start: $(date -Is)"
nvidia-smi
echo "=========================================="

# HF token is sourced at runtime from the existing run script (kept out of this file).
export HF_TOKEN="$(grep -oE 'hf_[A-Za-z0-9]+' /projects/adsum-mllm-om/mllm/qwen_pipeline/training/runs/run_sft_lora_h200.sh | head -1)"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_HOME=/projects/adsum-mllm-om/.hf_cache
export HF_HUB_CACHE=/projects/adsum-mllm-om/.hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /projects/adsum-mllm-om/mllm/qwen_pipeline
mkdir -p logs training/runs

# Trainer is trl's SFTTrainer (see train_lora.py). Install once, no-deps:
#   /projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python -m pip install --no-deps trl==0.24.0
SECONDS=0
/projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python training/train_lora.py \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --csv   data/train.csv \
  --jsonl data/train.jsonl \
  --eval_csv   data/eval.csv \
  --eval_jsonl data/eval.jsonl \
  --test_csv   data/test_final.csv \
  --test_jsonl data/test_final.jsonl \
  --h5    keyframes/keyframes_k5.h5 \
  --output_dir training/runs/qwen3vl8b_lora_r32 \
  --lora_r 32 \
  --fps 5 --attn_impl flash_attention_2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --logging_steps 5

echo "End: $(date -Is)  ELAPSED: ${SECONDS}s"
