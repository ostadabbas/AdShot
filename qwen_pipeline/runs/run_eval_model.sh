#!/bin/bash
#============================================================================
# Real-metric eval for ONE trained LoRA run: merge adapter -> generation eval
# on data/test.csv (124 rows) via keyframes_k5.h5. Produces the real pairing
# metrics (accuracy/precision/recall/f1/iou), comparable to results/base_test.*
# and results/sft_test.* (v1).
#
# Usage:
#   sbatch runs/run_eval_model.sh <adapter_dir> <result_prefix> [test_csv]
# Example:
#   sbatch runs/run_eval_model.sh training/runs/qwen3vl8b_lora_r16 r16
#   -> merges to training/runs/qwen3vl8b_lora_r16_merged
#   -> writes results/r16_test.{json,csv}
# test_csv defaults to data/test.csv (the 124-row set used for base/v1/sweep).
#============================================================================
#SBATCH --job-name=eval_model
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=80GB
#SBATCH --output=logs/eval_model_%j.out
#SBATCH --error=logs/eval_model_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rastogi.o@northeastern.edu

set -uo pipefail

ADAPTER="${1:?usage: sbatch run_eval_model.sh <adapter_dir> <result_prefix> [test_csv]}"
PREFIX="${2:?usage: sbatch run_eval_model.sh <adapter_dir> <result_prefix> [test_csv]}"
TEST_CSV="${3:-data/test.csv}"
MERGED="${ADAPTER}_merged"

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-interactive}  Node: ${SLURM_NODELIST:-$(hostname)}"
echo "adapter=$ADAPTER  prefix=$PREFIX  test_csv=$TEST_CSV"
echo "Start: $(date)"
nvidia-smi
echo "=========================================="

# Base model + adapter are local/cached; no HF token needed. If a private pull
# is ever required, `export HF_TOKEN=...` in your shell before sbatch.
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

if [ ! -d "$MERGED" ]; then
  echo "[merge] $ADAPTER -> $MERGED"
  $PY training/merge_adapter.py --base "$BASE" --adapter "$ADAPTER" --out "$MERGED" || exit 1
else
  echo "[merge] $MERGED already exists, skipping merge"
fi

echo "[eval] $MERGED on $TEST_CSV"
$PY keyframes/kf_batch_infer.py \
  --model "$MERGED" \
  --csv "$TEST_CSV" \
  --h5 keyframes/keyframes_k5.h5 \
  --output_json "results/${PREFIX}_test.json" \
  --output_csv  "results/${PREFIX}_test.csv" \
  --fps 5 --attn_impl flash_attention_2 \
  --batch_size 8 --num_workers 16 || exit 1

echo "[done] $PREFIX -> results/${PREFIX}_test.{json,csv}"
echo "End: $(date)"
curl -s -d "eval_model done: $PREFIX on $(hostname)" ntfy.sh/claude-om-notify >/dev/null || true
