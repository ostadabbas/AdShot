#!/bin/bash
#============================================================================
# SLURM Job Configuration - NVILA-Lite-8B-hf-preview Evaluation
#============================================================================
#SBATCH --job-name=nvila_lite8b_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --mem=80GB              
#SBATCH --output=logs/eval_nvila_lite_%j.out
#SBATCH --error=logs/eval_nvila_lite_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=we.xie@northeastern.edu

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

# Set up HuggingFace cache (suppress TRANSFORMERS_CACHE warning)
: "${HF_TOKEN:?export HF_TOKEN before running}"   # token comes from your environment
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
mkdir -p /projects/adsum-mllm-om/.hf_cache
export HF_HOME=/projects/adsum-mllm-om/.hf_cache
export TRANSFORMERS_CACHE=/projects/adsum-mllm-om/.hf_cache/transformers
export HF_HUB_DISABLE_XET=1

cd /projects/adsum-mllm-om/mllm/evaluations

# Create logs directory if needed
mkdir -p logs

# Activate environment
source /projects/adsum-mllm-om/miniconda3/bin/activate /projects/adsum-mllm-om/miniconda3/envs/mllm

echo "Starting NVILA-Lite-8B-hf-preview evaluation..."
echo "Model: Efficient-Large-Model/NVILA-Lite-8B-hf-preview"
echo "Using 5 frames per shot (optimized for NVILA efficiency)"
echo ""

echo "Test run complete. Full evaluation starting..."

# Full evaluation - 5fps (NVILA sweet spot)
python evaluate2_wen_nvila.py \
    --csv /projects/adsum-mllm-om/data/fullmatch.csv \
    --model Efficient-Large-Model/NVILA-Lite-8B-hf-preview \
    --output_json results_nvila_lite_5fps.json \
    --output_csv results_nvila_lite_5fps.csv \
    --fps 5 \
    --max_pixels 50176

# Conservative 3fps backup (if 5fps has memory issues)
python evaluate2_wen_nvila.py \
    --csv /projects/adsum-mllm-om/data/fullmatch.csv \
    --model Efficient-Large-Model/NVILA-Lite-8B-hf-preview \
    --output_json results_nvila_lite_2fps.json \
    --output_csv results_nvila_lite_2fps.csv \
    --fps 2 \
    --max_pixels 50176

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
    echo "✓ NVILA-Lite-8B evaluation completed successfully!"
    echo "Results saved to:"
    echo "  - results_nvila_lite_5fps.json"
    echo "  - results_nvila_lite_5fps.csv" 
    echo "  - results_nvila_lite_3fps.json"
    echo "  - results_nvila_lite_3fps.csv"
else
    echo "✗ Job failed with exit code $EXIT_CODE"
    echo "Check error log: logs/eval_nvila_lite_${SLURM_JOB_ID}.err"
    exit $EXIT_CODE
fi
