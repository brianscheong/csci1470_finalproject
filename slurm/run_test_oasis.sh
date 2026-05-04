#!/bin/bash

#SBATCH -J vxm_oasis_test
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o test_oasis.out
#SBATCH -e test_oasis.err

# =========================================
# Evaluates the trained OASIS VoxelMorph model on the test set.
# Edit --model and --out below to match the run you want to evaluate.
# Submit with: sbatch slurm/run_test_oasis.sh
# =========================================

echo "=============================="
echo "Job started on $(hostname)"
echo "Time: $(date)"
echo "=============================="

module load cuda/11.8
module load cudnn/8

PYTHON=/users/bcheong/.conda/envs/voxelmorph_tf/bin/python

echo "=============================="
echo "Starting OASIS evaluation..."
echo "=============================="

# --- VoxelMorph evaluation ---
$PYTHON -u scripts/test_oasis.py \
    --model weights/oasis_vxm_exp1_full_best.h5 \
    --out   results/dice_oasis_vxm.txt \
    --num_pairs 100

# --- No-warp baseline ---
$PYTHON -u scripts/test_oasis.py \
    --model none \
    --out   results/dice_oasis_nowarp.txt \
    --num_pairs 100

echo "=============================="
echo "Job finished at $(date)"
echo "=============================="
