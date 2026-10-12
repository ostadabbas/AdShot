#!/bin/bash
#SBATCH --job-name=pairext_batch12
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --mem=100GB
#SBATCH --output=logs/pairext_batch12_%j.out
#SBATCH --error=logs/pairext_batch12_%j.err
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rastogi.o@northeastern.edu

# Pair-extraction pipeline for batch12, isolated under data2/batch12.
# Every stage is resumable -- if this job hits the 8h wall, just resubmit:
#   sbatch runs/sbatch_batch12.sh
set -euo pipefail
export PYTHONUNBUFFERED=1                       # real-time (unbuffered) logs
export TORCH_HOME=/projects/adsum-mllm-om/.hf_cache/torch

BATCH=12
PY=/projects/adsum-mllm-om/miniconda3/envs/shot/bin/python
RAW_CSV=/projects/adsum-mllm-om/data2/batch${BATCH}_videos.csv
CSV=/projects/adsum-mllm-om/data2/batch${BATCH}_videos_fix.csv
OUT=/projects/adsum-mllm-om/data2/batch${BATCH}

cd /projects/adsum-mllm-om/mllm/pair_extraction
mkdir -p logs "$OUT"
echo "==== batch ${BATCH} | job ${SLURM_JOB_ID} | $(date) ===="; nvidia-smi

# Stage 0: brand_id normalize (b1306->1306) + sanitize '/' in brand_name + dedup ad_id.
# Idempotent. Use raw CSV when present, else fix the existing _fix.csv in place.
if [ -f "$RAW_CSV" ]; then
    $PY s0_fix_brand_id.py --input "$RAW_CSV" --output "$CSV"
else
    $PY s0_fix_brand_id.py --input "$CSV" --output "$CSV"
fi

# Stage 1a: shot detection -> shotsplits (resumable: skips ad_ids already done)
$PY s1_shot_detection.py --data_csv "$CSV" --output_dir "$OUT/shotsplits" \
    --num_decode_workers 24 --num_gpu_workers 1

# Stage 1b+1c: fused keyframe+embedding -> embeddings (resumable; writes no keyframe files)
$PY s1bc_fused.py --data_csv "$CSV" --shotsplits_dir "$OUT/shotsplits" \
    --output_dir "$OUT/embeddings" --num_decode_workers 24

# Stage 2: clustering -> clusters (resumable; keyframes_dir is an unused required arg)
$PY s2_clustering.py --embeddings_dir "$OUT/embeddings" \
    --keyframes_dir "$OUT/keyframes" --output_dir "$OUT/clusters"

# Stage 3: pairing -> fullmatch.csv (min_mapping_degree=0.9 default)
$PY s3_pairing.py --shotsplits_dir "$OUT/shotsplits" --clusters_dir "$OUT/clusters" \
    --data_csv "$CSV" --out_path "$OUT/fullmatch.csv"

# Stage 4: pair viz (decodes p50 frames straight from video; no keyframes dir)
$PY pair_viz.py --fullmatch_csv "$OUT/fullmatch.csv" --output_dir "$OUT/pair_viz"

echo "==== batch ${BATCH} DONE | $(date) ===="
