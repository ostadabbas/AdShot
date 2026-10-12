#!/bin/bash
# Run batch1 through every stage, one script at a time.
# Stage-1 outputs go to the config dirs (data2/0.1/{shotsplits,embeddings}); the fused
# 1b+1c stage writes no keyframe files. Each stage reads the previous stage's output and
# stages 2/3 find them by default. Every stage is resumable -- safe to stop and restart.
set -e

export TORCH_HOME=/projects/adsum-mllm-om/.hf_cache/torch
cd /projects/adsum-mllm-om/mllm/pair_extraction

# Stage 1a: shot detection (24 CPU decode workers, threads=1 each -> 1 GPU worker)
/projects/adsum-mllm-om/miniconda3/envs/shot/bin/python s1_shot_detection.py \
  --data_csv /projects/adsum-mllm-om/data2/batch1_videos_fix.csv \
  --num_decode_workers 24 --num_gpu_workers 1

# Stage 1b+1c: fused keyframe extraction + embeddings.
# No keyframe files written (avoids the NFS small-file storm); 24 CPU decode workers
# feed 1 GPU embed worker through a bounded queue, so decode overlaps the Swin3D pass.
# Reads shotsplits and writes embeddings to the config dirs by default.
/projects/adsum-mllm-om/miniconda3/envs/shot/bin/python s1bc_fused.py \
  --data_csv /projects/adsum-mllm-om/data2/batch1_videos_fix.csv \
  --num_decode_workers 24

# Stage 2: clustering
/projects/adsum-mllm-om/miniconda3/envs/shot/bin/python s2_clustering.py

# Stage 3: pairing (writes config FULLMATCH_OUT = data2/fullmatch.csv)
/projects/adsum-mllm-om/miniconda3/envs/shot/bin/python s3_pairing.py \
  --data_csv /projects/adsum-mllm-om/data2/batch1_videos_fix.csv

# Stage 4: pair viz (QA). Decodes p50 frames straight from the source videos using
# the shotsplit + relative_path columns in fullmatch.csv -- no keyframes dir needed
# (s1bc_fused writes none).
/projects/adsum-mllm-om/miniconda3/envs/shot/bin/python pair_viz.py \
  --fullmatch_csv /projects/adsum-mllm-om/data2/fullmatch.csv \
  --output_dir /projects/adsum-mllm-om/data2/0.1/pair_viz
