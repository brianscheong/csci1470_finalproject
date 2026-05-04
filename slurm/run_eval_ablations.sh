#!/bin/bash

# =========================================
# Evaluates all trained ablation models on the OASIS test set.
# Runs sequentially in one job (each eval is ~10-30 min).
# Submit with: sbatch slurm/run_eval_ablations.sh
# =========================================

#SBATCH -J vxm_eval_ablations
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o eval_ablations.out
#SBATCH -e eval_ablations.err

PYTHON=/users/bcheong/.conda/envs/voxelmorph_tf/bin/python

module load cuda/11.8
module load cudnn/8

echo "=============================="
echo "Ablation evaluation started"
echo "Time: $(date)"
echo "=============================="

# --- Experiment 3: Lambda ablation ---
for LAMBDA in 0.0 0.1 0.5 1.0; do
    TAG="lambda_${LAMBDA}"
    MODEL="weights/oasis_vxm_${TAG}_best.h5"
    OUT="results/dice_oasis_${TAG}.txt"
    echo "Evaluating ${TAG}..."
    $PYTHON -u scripts/test_oasis.py --model $MODEL --out $OUT --num_pairs 100
done

# --- Experiment 4: Loss ablation ---
for LOSS in mse ncc ssim; do
    TAG="loss_${LOSS}"
    MODEL="weights/oasis_vxm_${TAG}_best.h5"
    OUT="results/dice_oasis_${TAG}.txt"
    echo "Evaluating ${TAG}..."
    $PYTHON -u scripts/test_oasis.py --model $MODEL --out $OUT --num_pairs 100
done

# --- Experiment 2: Training size ablation ---
for SIZE in 10 100 full; do
    TAG="size_${SIZE}"
    MODEL="weights/oasis_vxm_${TAG}_best.h5"
    OUT="results/dice_oasis_${TAG}.txt"
    echo "Evaluating ${TAG}..."
    $PYTHON -u scripts/test_oasis.py --model $MODEL --out $OUT --num_pairs 100
done

echo "=============================="
echo "All evaluations done at $(date)"
echo "=============================="
