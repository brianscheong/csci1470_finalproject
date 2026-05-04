#!/bin/bash

# =========================================
# Experiment 3: Smoothness Regularization Ablation
# Trains VoxelMorph on OASIS with four different lambda values:
#   lambda in {0, 0.1, 0.5, 1.0}
# Each run gets its own SLURM job so they can run in parallel.
#
# Usage (from project root):
#   bash slurm/run_ablation_lambda.sh
# =========================================

PYTHON=/users/bcheong/.conda/envs/voxelmorph_tf/bin/python

for LAMBDA in 0.0 0.1 0.5 1.0; do

    TAG="lambda_${LAMBDA}"
    OUT_FILE="train_lambda_${LAMBDA}.out"
    ERR_FILE="train_lambda_${LAMBDA}.err"

    sbatch <<EOF
#!/bin/bash
#SBATCH -J vxm_${TAG}
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o ${OUT_FILE}
#SBATCH -e ${ERR_FILE}

echo "=============================="
echo "Lambda ablation: GRAD_WEIGHT=${LAMBDA}"
echo "Job started on \$(hostname)"
echo "Time: \$(date)"
echo "=============================="

module load cuda/11.8
module load cudnn/8

GRAD_WEIGHT=${LAMBDA} LOSS_TYPE=ncc RUN_TAG=${TAG} \\
    ${PYTHON} -u scripts/train_oasis.py

echo "=============================="
echo "Job finished at \$(date)"
echo "=============================="
EOF

    echo "Submitted job for GRAD_WEIGHT=${LAMBDA} (tag: ${TAG})"
done
