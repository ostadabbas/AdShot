#!/bin/bash
#============================================================================
# Batched Qwen3-VL-8B eval on the TEST split at a configurable frames-per-shot.
# Submit once per fps:  sbatch --export=ALL,FPS=N --job-name=... --output=... this
# Each fps writes its own results file and its own data/frames/k<fps>/ cache,
# so the three jobs are independent and safe to run concurrently.
#============================================================================
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=80GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rastogi.o@northeastern.edu

set -euo pipefail

FPS="${FPS:?set FPS via --export=ALL,FPS=N}"

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-interactive}  Node: ${SLURM_NODELIST:-$(hostname)}"
echo "FPS (frames-per-shot): ${FPS}"
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

cd /projects/adsum-mllm-om/mllm/qwen_pipeline
mkdir -p logs results

PY=/projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python

SECONDS=0
$PY batch_infer.py \
  --csv data/test.csv \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --output_json results/qwen3vl_8b_test_${FPS}fps.json \
  --output_csv  results/qwen3vl_8b_test_${FPS}fps.csv \
  --fps "${FPS}" --batch_size 16 --num_workers 16

echo "End: $(date -Is)  ELAPSED: ${SECONDS}s"
