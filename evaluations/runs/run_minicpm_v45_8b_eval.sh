#!/bin/bash
#============================================================================
# SLURM Job Configuration
#============================================================================
#SBATCH --job-name=minicpm_v45_8B_fps1_fps3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --mem=80GB              
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
echo "Node: $SLURM_NODELIST"
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
source /projects/adsum-mllm-om/miniconda3/bin/activate /projects/adsum-mllm-om/miniconda3/envs/mllm3

# Create logs directory if needed
mkdir -p logs
mkdir -p new_results

echo "Starting evaluation 5fps..."
python evaluate3.py \
     --csv /projects/adsum-mllm-om/data/fullmatch823.csv \
     --model openbmb/MiniCPM-V-4_5 \
     --output_json new_results/minicpm_v45_8b_5fps.json \
     --output_csv new_results/minicpm_v45_8b_5fps.csv \
     --fps 5

echo "Starting evaluation 1fps..."
python evaluate3.py \
     --csv /projects/adsum-mllm-om/data/fullmatch823.csv \
     --model openbmb/MiniCPM-V-4_5 \
     --output_json new_results/minicpm_v45_8b_1fps.json \
     --output_csv new_results/minicpm_v45_8b_1fps.csv \
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