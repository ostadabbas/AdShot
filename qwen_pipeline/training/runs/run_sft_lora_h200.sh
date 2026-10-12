#!/bin/bash
#============================================================================
# LoRA SFT of Qwen3-VL-8B on the shot-selection task, on a DEDICATED H200.
# Frames are read from keyframes/keyframes_k5.h5 (no ffmpeg, no NFS hot path).
# Mirrors qwen_pipeline/runs/run_qwen3vl_8b_fullmatch_h5.sh.
#============================================================================
#SBATCH --job-name=qwen3vl8b_sft_lora
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --mem=96GB
#SBATCH --output=logs/sft_lora_%j.out
#SBATCH --error=logs/sft_lora_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rastogi.o@northeastern.edu

set -euo pipefail

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-interactive}  Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Start: $(date -Is)"
nvidia-smi
echo "=========================================="

: "${HF_TOKEN:?export HF_TOKEN before running}"   # token comes from your environment
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_HOME=/projects/adsum-mllm-om/.hf_cache
export HF_HUB_CACHE=/projects/adsum-mllm-om/.hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /projects/adsum-mllm-om/mllm/qwen_pipeline
mkdir -p logs training/runs

# Trainer is now trl's SFTTrainer (see train_lora.py). Install once, no-deps so the
# pinned transformers 4.57.5 + flash-attn2 stack is untouched:
#   /projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python -m pip install --no-deps trl==0.24.0
# Fresh output dir (v2) keeps the transformers.Trainer v1 adapter/results for A/B compare.
SECONDS=0
/projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python training/train_lora.py \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --csv   data/train.csv \
  --jsonl data/train.jsonl \
  --eval_csv   data/test.csv \
  --eval_jsonl data/test.jsonl \
  --h5    keyframes/keyframes_k5.h5 \
  --output_dir training/runs/qwen3vl8b_lora_v2 \
  --fps 5 --attn_impl flash_attention_2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --logging_steps 5

echo "End: $(date -Is)  ELAPSED: ${SECONDS}s"
