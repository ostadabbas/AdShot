#!/bin/bash
#============================================================================
# SMOKE TEST: validate flash_attention_2 + dynamic batching on the HDF5 eval.
# Writes to results/SMOKE_*.json/csv (does NOT touch the real fullmatch results).
# Purpose: confirm the numel-overflow / CUDA-OOM errors are gone.
#============================================================================
#SBATCH --job-name=qwen3vl8b_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:40:00
#SBATCH --mem=80GB
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rastogi.o@northeastern.edu

set -euo pipefail

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-interactive}  Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Start: $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=========================================="

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
# --limit 200 covers the rows where the big (>200-frame) samples live, so the
# numel/OOM path is actually exercised. Fresh output -> processes all 200.
$PY keyframes/kf_batch_infer.py \
  --csv data/fullmatch_all.csv \
  --h5  keyframes/keyframes_k5.h5 \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --output_json results/SMOKE_qwen3vl_8b_5fps.json \
  --output_csv  results/SMOKE_qwen3vl_8b_5fps.csv \
  --fps 5 --attn_impl flash_attention_2 \
  --batch_size 8 --max_frames_per_batch 480 --num_workers 8 \
  --limit 200

echo "End: $(date -Is)  ELAPSED: ${SECONDS}s"
echo "=== error count in smoke results ==="
$PY - <<'PYEOF'
import csv
rows=list(csv.DictReader(open("results/SMOKE_qwen3vl_8b_5fps.csv")))
errs=[r for r in rows if r.get("error")]
print(f"rows={len(rows)} errors={len(errs)}")
for r in errs[:10]: print(" ERR:", r.get("error","")[:100])
PYEOF