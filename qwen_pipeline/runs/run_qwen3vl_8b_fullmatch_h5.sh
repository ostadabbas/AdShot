#!/bin/bash
#============================================================================
# Full Qwen3-VL-8B eval on fullmatch_all, reading keyframes from the HDF5
# container (keyframes/keyframes_k5.h5). Runs on a DEDICATED SLURM GPU node —
# the interactive node was shared and caused CUDA OOM.
#============================================================================
#SBATCH --job-name=qwen3vl8b_fullmatch_h5
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --mem=96GB
#SBATCH --output=logs/fullmatch_h5_%j.out
#SBATCH --error=logs/fullmatch_h5_%j.err
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
export HF_DATASETS_CACHE=/projects/adsum-mllm-om/.hf_cache/datasets
export TRANSFORMERS_CACHE=/projects/adsum-mllm-om/.hf_cache/transformers
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /projects/adsum-mllm-om/mllm/qwen_pipeline
mkdir -p logs results

PY=/projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python

SECONDS=0
$PY keyframes/kf_batch_infer.py \
  --csv data/fullmatch_all.csv \
  --h5  keyframes/keyframes_k5.h5 \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --output_json results/qwen3vl_8b_fullmatch_5fps.json \
  --output_csv  results/qwen3vl_8b_fullmatch_5fps.csv \
  --fps 5 --attn_impl flash_attention_2 \
  --batch_size 8 --max_frames_per_batch 480 --num_workers 8

echo "End: $(date -Is)  ELAPSED: ${SECONDS}s"
