#!/bin/bash

#SBATCH -J vxm_test
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o test.out
#SBATCH -e test.err

# =========================================
# this is a shell script to run the test_kits.py script on the cluster. It uses SLURM for job scheduling.
# You can submit this script using "sbatch run_test_kits.sh" from the terminal. It allows me to run it without maintaining an active ssh connection to the cluster
# for the full several hours that it takes.
# terminal output will be saved to train.out and train.err files. You can check these files to monitor progress and debug as needed.
# =========================================

echo "=============================="
echo "Job started on $(hostname)"
echo "Time: $(date)"
echo "=============================="

# --- Load CUDA (required for GPU) ---
module load cuda/11.8
module load cudnn/8
conda init
conda activate voxelmorph_tf
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# --- Use your exact environment python ---
PYTHON=/users/bcheong/.conda/envs/voxelmorph_tf/bin/python

echo "=============================="
echo "Starting testing..."
echo "=============================="

# --- Run testing ---
$PYTHON -u scripts/test_kits.py

echo "=============================="
echo "Job finished at $(date)"
echo "=============================="