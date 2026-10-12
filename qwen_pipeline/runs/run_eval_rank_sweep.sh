#!/bin/bash
#============================================================================
# Merge + real-metric eval for the FULL completed LoRA rank sweep
# (r16/r32/r64/r128/r256/r512). For each rank: merge adapter -> standalone dir,
# then run kf_batch_infer.py on data/test.csv (124 rows) via keyframes_k5.h5.
# Produces the real pairing metrics (accuracy/precision/recall/f1/iou) ->
# results/r<N>_test.{json,csv}, comparable to results/base_test.* and
# results/sft_test.* (v1).
#
# Resume-safe: skips ranks whose results/r<N>_test.json already exists, and
# skips merging when the *_merged dir already exists. Safe to resubmit.
#============================================================================
#SBATCH --job-name=eval_rank_sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --mem=80GB
#SBATCH --output=logs/eval_rank_sweep_%j.out
#SBATCH --error=logs/eval_rank_sweep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rastogi.o@northeastern.edu

set -uo pipefail

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-interactive}  Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Start: $(date)"
nvidia-smi
echo "=========================================="

# Base model + adapters are local/cached; no HF token needed. If a private
# pull is ever required, `export HF_TOKEN=...` in your shell before sbatch.
export HF_HOME=/projects/adsum-mllm-om/.hf_cache
export HF_HUB_CACHE=/projects/adsum-mllm-om/.hf_cache/hub
export HF_DATASETS_CACHE=/projects/adsum-mllm-om/.hf_cache/datasets
export TRANSFORMERS_CACHE=/projects/adsum-mllm-om/.hf_cache/transformers
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1

cd /projects/adsum-mllm-om/mllm/qwen_pipeline
mkdir -p logs results

PY=/projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python
BASE="Qwen/Qwen3-VL-8B-Instruct"

for R in r16 r32 r64 r128 r256 r512; do
  ADAPTER="training/runs/qwen3vl8b_lora_${R}"
  MERGED="training/runs/qwen3vl8b_lora_${R}_merged"
  echo ""
  echo "######### ${R} #########  $(date)"

  if [ -f "results/${R}_test.json" ]; then
    echo "[skip] results/${R}_test.json already exists, skipping $R"
    continue
  fi

  if [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
    echo "[skip] adapter missing for $R ($ADAPTER), skipping"
    continue
  fi

  if [ ! -d "$MERGED" ]; then
    echo "[merge] $ADAPTER -> $MERGED"
    $PY training/merge_adapter.py --base "$BASE" --adapter "$ADAPTER" --out "$MERGED" \
      || { echo "[merge] FAILED for $R"; continue; }
  else
    echo "[merge] $MERGED already exists, skipping merge"
  fi

  echo "[eval] $MERGED on data/test.csv"
  $PY keyframes/kf_batch_infer.py \
    --model "$MERGED" \
    --csv data/test.csv \
    --h5 keyframes/keyframes_k5.h5 \
    --output_json "results/${R}_test.json" \
    --output_csv  "results/${R}_test.csv" \
    --fps 5 --attn_impl flash_attention_2 \
    --batch_size 8 --num_workers 16 \
    || { echo "[eval] FAILED for $R"; continue; }

  echo "[done] $R -> results/${R}_test.{json,csv}"
done

echo ""
echo "End: $(date)"
curl -s -d "rank-sweep eval done (r16/r32/r64/r128/r256/r512) on $(hostname)" ntfy.sh/claude-om-notify >/dev/null || true
