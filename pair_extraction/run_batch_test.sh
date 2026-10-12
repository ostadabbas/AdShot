#!/bin/bash
# batch_test (7 brands) through every stage, one script at a time.
# All output goes to the scratch dir data2/0.1_test (does NOT touch the live batch1
# run in data2/0.1). Every stage is resumable. Run the whole file, or copy one stage.
set -e

export TORCH_HOME=/projects/adsum-mllm-om/.hf_cache/torch
cd /projects/adsum-mllm-om/mllm/pair_extraction

# build the 7-brand subset CSV
/projects/adsum-mllm-om/miniconda3/envs/shot/bin/python make_batch_test.py

# Stage 1a: shot detection (24 CPU decode workers -> 1 GPU worker)
/projects/adsum-mllm-om/miniconda3/envs/shot/bin/python s1_shot_detection.py \
  --data_csv /projects/adsum-mllm-om/data2/batch_test_videos_fix.csv \
  --output_dir /projects/adsum-mllm-om/data2/0.1_test/shotsplits \
  --num_decode_workers 24 --num_gpu_workers 1

# Stage 1b: keyframes (24 workers -- was the slow part at the default 8)
/projects/adsum-mllm-om/miniconda3/envs/shot/bin/python s1_keyframes.py \
  --data_csv /projects/adsum-mllm-om/data2/batch_test_videos_fix.csv \
  --shotsplits_dir /projects/adsum-mllm-om/data2/0.1_test/shotsplits \
  --output_dir /projects/adsum-mllm-om/data2/0.1_test/keyframes \
  --num_workers 24

# Stage 1c: embeddings (Swin3D-B loaded once)
/projects/adsum-mllm-om/miniconda3/envs/shot/bin/python s1_embedding.py \
  --data_csv /projects/adsum-mllm-om/data2/batch_test_videos_fix.csv \
  --keyframes_dir /projects/adsum-mllm-om/data2/0.1_test/keyframes \
  --output_dir /projects/adsum-mllm-om/data2/0.1_test/embeddings

# Stage 2: clustering
/projects/adsum-mllm-om/miniconda3/envs/shot/bin/python s2_clustering.py \
  --embeddings_dir /projects/adsum-mllm-om/data2/0.1_test/embeddings \
  --keyframes_dir /projects/adsum-mllm-om/data2/0.1_test/keyframes \
  --output_dir /projects/adsum-mllm-om/data2/0.1_test/clusters

# Stage 3: pairing
/projects/adsum-mllm-om/miniconda3/envs/shot/bin/python s3_pairing.py \
  --shotsplits_dir /projects/adsum-mllm-om/data2/0.1_test/shotsplits \
  --clusters_dir /projects/adsum-mllm-om/data2/0.1_test/clusters \
  --data_csv /projects/adsum-mllm-om/data2/batch_test_videos_fix.csv \
  --out_path /projects/adsum-mllm-om/data2/0.1_test/fullmatch.csv

# Stage 4: pair viz (QA). Decodes p50 frames straight from the source videos using
# the shotsplit + relative_path columns in fullmatch.csv -- no keyframes dir needed.
/projects/adsum-mllm-om/miniconda3/envs/shot/bin/python pair_viz.py \
  --fullmatch_csv /projects/adsum-mllm-om/data2/0.1_test/fullmatch.csv \
  --output_dir /projects/adsum-mllm-om/data2/0.1_test/pair_viz
