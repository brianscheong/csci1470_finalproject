#!/bin/bash

# =========================================
# Experiment 2: Training Set Size
# Trains VoxelMorph on OASIS using 10, 100, and the full training set.
# Each run gets its own SLURM job.
#
# Usage (from project root):
#   bash slurm/run_trainsize_sweep.sh
# =========================================

PYTHON=/users/bcheong/.conda/envs/voxelmorph_tf/bin/python

for SIZE in 10 100 full; do

    TAG="size_${SIZE}"
    OUT_FILE="train_size_${SIZE}.out"
    ERR_FILE="train_size_${SIZE}.err"

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
echo "Train size sweep: TRAIN_SIZE=${SIZE}"
echo "Job started on \$(hostname)"
echo "Time: \$(date)"
echo "=============================="

module load cuda/11.8
module load cudnn/8

TRAIN_SIZE=${SIZE} GRAD_WEIGHT=1.0 LOSS_TYPE=ncc RUN_TAG=${TAG} \\
    ${PYTHON} -u scripts/train_oasis.py

echo "=============================="
echo "Job finished at \$(date)"
echo "=============================="
EOF

    echo "Submitted job for TRAIN_SIZE=${SIZE} (tag: ${TAG})"
done
