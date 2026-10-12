#!/bin/bash
#============================================================================
# Full pair_extraction pipeline on the 7-brand batch_test, end-to-end, on one
# a100. Wraps run_batch_test.sh (set -e, env `shot`) which writes to the scratch
# dir data2/0.1_test. Used to PROVE the archived/removed files are unreachable:
# if this completes green with archive/ moved out, the removals are safe.
#============================================================================
#SBATCH --job-name=pe_batch_test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=80GB
#SBATCH --output=logs/pe_batch_test_%j.out
#SBATCH --error=logs/pe_batch_test_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rastogi.o@northeastern.edu

set -uo pipefail

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-interactive}  Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Start: $(date)"
nvidia-smi
echo "=========================================="

cd /projects/adsum-mllm-om/mllm/pair_extraction
mkdir -p logs

echo "[run] bash run_batch_test.sh"
bash run_batch_test.sh
rc=$?

echo "=========================================="
echo "run_batch_test.sh exit code: $rc"
echo "End: $(date)"
if [ $rc -eq 0 ]; then
  echo "[result] full batch_test pipeline PASSED"
  curl -s -d "pe batch_test PASSED (archive removals safe) on $(hostname)" ntfy.sh/claude-om-notify >/dev/null || true
else
  echo "[result] batch_test pipeline FAILED (rc=$rc) -- check .err"
  curl -s -d "pe batch_test FAILED rc=$rc on $(hostname)" ntfy.sh/claude-om-notify >/dev/null || true
fi
exit $rc
