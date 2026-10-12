#!/bin/bash
#============================================================================
# SLURM Job Configuration
#============================================================================
#SBATCH --job-name=llava-onevision-qwen2-7b_fps1_fps5
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --mem=40GB              
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rastogi.o@northeastern.edu

set -euo pipefail

#============================================================================
# Print Job Information
#============================================================================
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="
echo "GPU Information:"
nvidia-smi
echo "=========================================="

# Go to working directory
: "${HF_TOKEN:?export HF_TOKEN before running}"   # token comes from your environment
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
mkdir -p /projects/adsum-mllm-om/.hf_cache
export HF_HOME=/projects/adsum-mllm-om/.hf_cache
export HF_HUB_CACHE=/projects/adsum-mllm-om/.hf_cache/hub
export HF_DATASETS_CACHE=/projects/adsum-mllm-om/.hf_cache/datasets
export TRANSFORMERS_CACHE=/projects/adsum-mllm-om/.hf_cache/transformers
export HF_HUB_DISABLE_XET=1

nvidia-smi
cd /projects/adsum-mllm-om/mllm/evaluations
source /projects/adsum-mllm-om/miniconda3/bin/activate /projects/adsum-mllm-om/miniconda3/envs/mllm2

# Create logs directory if needed
mkdir -p logs
mkdir -p new_results

echo "Starting evaluation 5fps..."
/projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python \
    evaluate2.py \
     --csv /projects/adsum-mllm-om/data/fullmatch823.csv \
     --model llava-hf/llava-onevision-qwen2-7b-ov-hf \
     --output_json new_results/llava-onevision-qwen2-7b_5fps.json \
     --output_csv new_results/llava-onevision-qwen2-7b_5fps.csv \
     --attn_impl flash_attention_2 \
     --fps 5

echo "Starting evaluation 1fps..."
/projects/adsum-mllm-om/miniconda3/envs/mllm2/bin/python \
    evaluate2.py \
     --csv /projects/adsum-mllm-om/data/fullmatch823.csv \
     --model llava-hf/llava-onevision-qwen2-7b-ov-hf \
     --output_json new_results/llava-onevision-qwen2-7b_1fps.json \
     --output_csv new_results/llava-onevision-qwen2-7b_1fps.csv \
     --attn_impl flash_attention_2 \
     --fps 1

#============================================================================
# Check Exit Status and Report
#============================================================================
EXIT_CODE=$?
echo ""
echo "=========================================="
echo "Python script exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Job completed successfully!"
else
    echo "✗ Job failed with exit code $EXIT_CODE"
    echo "Check error log: logs/eval_${SLURM_JOB_ID}.err"
    exit $EXIT_CODE
fi