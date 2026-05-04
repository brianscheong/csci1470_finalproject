#!/bin/bash

# =========================================
# Experiment 4: Image Similarity Loss Ablation
# Trains VoxelMorph on OASIS with three different similarity losses:
#   MSE, NCC (CC), SSIM
# Each run gets its own SLURM job.
#
# Usage (from project root):
#   bash slurm/run_ablation_loss.sh
# =========================================

PYTHON=/users/bcheong/.conda/envs/voxelmorph_tf/bin/python

for LOSS in mse ncc ssim; do

    TAG="loss_${LOSS}"
    OUT_FILE="train_loss_${LOSS}.out"
    ERR_FILE="train_loss_${LOSS}.err"

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
echo "Loss ablation: LOSS_TYPE=${LOSS}"
echo "Job started on \$(hostname)"
echo "Time: \$(date)"
echo "=============================="

module load cuda/11.8
module load cudnn/8

GRAD_WEIGHT=1.0 LOSS_TYPE=${LOSS} RUN_TAG=${TAG} \\
    ${PYTHON} -u scripts/train_oasis.py

echo "=============================="
echo "Job finished at \$(date)"
echo "=============================="
EOF

    echo "Submitted job for LOSS_TYPE=${LOSS} (tag: ${TAG})"
done
