#!/bin/bash
#============================================================================
# Batched Qwen3-VL-8B eval on the locked-0.8 fullmatch dataset (batch1-5).
# Mirrors evaluations/runs/run_qwen3vl_8b_eval.sh but uses qwen_pipeline.
#============================================================================
#SBATCH --job-name=qwen3vl_8b_batch_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --mem=80GB
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rastogi.o@northeastern.edu

set -euo pipefail

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-interactive}  Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Start: $(date)"
nvidia-smi
echo "=========================================="

: "${HF_TOKEN:?export HF_TOKEN before running}"   # token comes from your environment
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_HOME=/projects/adsum-mllm-om/.hf_cache
export HF_HUB_CACHE=/projects/adsum-mllm-om/.hf_cache/hub
export HF_DATASETS_CACHE=/projects/adsum-mllm-om/.hf_cache/datasets
export TRANSFORMERS_CACHE=/projects/adsum-mllm-om/.hf_cache/transformers
export HF_HUB_DISABLE_XET=1

cd /projects/adsum-mllm-om/mllm/qwen_pipeline
mkdir -p logs results

PY=/projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python

# Full test split, batched (fps=5).
$PY batch_infer.py \
  --csv data/test.csv \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --output_json results/qwen3vl_8b_test_5fps.json \
  --output_csv  results/qwen3vl_8b_test_5fps.csv \
  --fps 5 --batch_size 4 --num_workers 16

echo "End: $(date)"
