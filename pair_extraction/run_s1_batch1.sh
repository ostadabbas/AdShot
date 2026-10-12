#!/bin/bash
# Run ONLY shot detection (s1_shot_detection.py) on batch1.
export TORCH_HOME=/projects/adsum-mllm-om/.hf_cache/torch
cd /projects/adsum-mllm-om/mllm/pair_extraction

python s1_shot_detection.py \
  --data_csv /projects/adsum-mllm-om/data2/batch1_videos_fix.csv \
  --video_dir /projects/adsum-mllm-om/videomedia
