#!/bin/bash
#============================================================================
# DRY RUN — validate the full train_lora.py path end-to-end on a GPU with tiny
# data: 2 train rows, 1 epoch (so per-epoch eval fires once), then the final
# TEST eval. report_to none, throwaway output dir. Runs on the 1h `sharing`
# h200 so it schedules fast. NOT a real training run.
#============================================================================
#SBATCH --job-name=sft_dryrun
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --mem=64GB
#SBATCH --output=logs/sft_dryrun_%j.out
#SBATCH --error=logs/sft_dryrun_%j.err

set -euo pipefail

echo "=========================================="
echo "DRY RUN  Job ID: ${SLURM_JOB_ID:-interactive}  Node: ${SLURM_NODELIST:-$(hostname)}"
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
mkdir -p logs

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
  --output_dir training/runs/qwen3vl8b_lora_dryrun \
  --fps 5 --attn_impl flash_attention_2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --limit 2 --eval_limit 2 --test_limit 2 \
  --logging_steps 1 \
  --resume off \
  --report_to none

echo "DRY RUN End: $(date -Is)  ELAPSED: ${SECONDS}s"
