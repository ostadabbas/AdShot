#!/bin/bash
#SBATCH --job-name=pair_extraction
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --mem=40GB
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#
# Generic sbatch runner for the pair-extraction pipeline.
# Paths/defaults come from config.yaml; override on the CLI below.
#
# Usage:
#   sbatch runs/run_pipeline.sh                         # stage=all, default batch
#   STAGE=clustering BRAND_ID=975 sbatch runs/run_pipeline.sh
#   DATA_CSV=/path/to/batchN_videos_fix.csv sbatch runs/run_pipeline.sh

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Start time: $(date)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "no nvidia-smi"

# --- environment ---
PY=/projects/adsum-mllm-om/miniconda3/envs/shot/bin/python
export TORCH_HOME=/projects/adsum-mllm-om/.hf_cache/torch   # Swin3D weights

# Run from the package directory so `import config` resolves.
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PKG_DIR"
mkdir -p logs

# --- knobs (all optional; default to config.yaml) ---
STAGE="${STAGE:-all}"
EXTRA_ARGS=( --stage "$STAGE" )
[ -n "${DATA_CSV:-}" ]    && EXTRA_ARGS+=( --data_csv "$DATA_CSV" )
[ -n "${OUTPUT_DIR:-}" ]  && EXTRA_ARGS+=( --output_dir "$OUTPUT_DIR" )
[ -n "${BRAND_ID:-}" ]    && EXTRA_ARGS+=( --brand_id "$BRAND_ID" )
[ -n "${BRAND:-}" ]       && EXTRA_ARGS+=( --brand "$BRAND" )
[ -n "${THRESHOLD:-}" ]   && EXTRA_ARGS+=( --threshold "$THRESHOLD" )

echo "Starting pipeline: ${EXTRA_ARGS[*]}"
"$PY" run_pipeline.py "${EXTRA_ARGS[@]}"

EXIT_CODE=$?
echo "Python exit code: $EXIT_CODE"
echo "End time: $(date)"
exit $EXIT_CODE
