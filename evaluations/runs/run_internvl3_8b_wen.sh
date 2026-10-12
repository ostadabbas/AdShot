#!/bin/bash
#============================================================================
# SLURM Job Configuration - InternVL2.5-8B Evaluation
#============================================================================
#SBATCH --job-name=int38beval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --mem=40GB              
#SBATCH --output=/projects/adsum-mllm-om/logs/eval_internvl3_8b_%j.out
#SBATCH --error=/projects/adsum-mllm-om/logs/eval_internvl3_8b_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=we.xie@northeastern.edu

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

# Set up HuggingFace cache
: "${HF_TOKEN:?export HF_TOKEN before running}"   # token comes from your environment
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
mkdir -p /projects/adsum-mllm-om/.hf_cache
export HF_HOME=/projects/adsum-mllm-om/.hf_cache
export HF_HUB_CACHE=/projects/adsum-mllm-om/.hf_cache/hub
export HF_DATASETS_CACHE=/projects/adsum-mllm-om/.hf_cache/datasets
export TRANSFORMERS_CACHE=/projects/adsum-mllm-om/.hf_cache/transformers
export HF_HUB_DISABLE_XET=1

cd /projects/adsum-mllm-om/mllm/evaluations
nvidia-smi

# # Create logs directory if needed
# mkdir -p logs

# Activate environment (use mllm with transformers==4.45.0)
source /projects/adsum-mllm-om/miniconda3/bin/activate /projects/adsum-mllm-om/miniconda3/envs/mllm2

echo "Starting InternVL2.5-8B evaluation..."
echo "Note: Using 3 frames per shot (10%, 50%, 90%) for 16K context limit"
echo ""

# Run evaluation with InternVL-optimized settings
python evaluate2_wen.py \
    --csv /projects/adsum-mllm-om/data/fullmatch823.csv \
    --model OpenGVLab/InternVL3_5-8B \
    --output_json new_results/internvl35_8b_3fps.json \
    --output_csv new_results/internvl35_8b_3fps.csv \
    --fps 3
    
python evaluate2_wen.py \
    --csv /projects/adsum-mllm-om/data/fullmatch823.csv \
    --model OpenGVLab/InternVL3_5-8B \
    --output_json new_results/internvl35_8b_1fps.json \
    --output_csv new_results/internvl35_8b_1fps.csv \
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
    echo "Results saved to:"
    echo "  - results_internvl25_8b.json"
    echo "  - results_internvl25_8b.csv"
else
    echo "✗ Job failed with exit code $EXIT_CODE"
    echo "Check error log: logs/eval_internvl_${SLURM_JOB_ID}.err"
    exit $EXIT_CODE
fi