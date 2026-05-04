#!/bin/bash

#SBATCH -J preprocess_oasis
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o preprocess_oasis.out
#SBATCH -e preprocess_oasis.err

# =========================================
# Preprocesses OASIS brain MRI data: resample → normalize → crop/pad.
# Submit with: sbatch slurm/run_preprocess_oasis.sh
# =========================================

PYTHON=/oscar/rt/9.6/25/x86_64_v3/miniforge3-25.3.0-3-a6hhdjzejtacz63sugjqnvgosfqz63ul/bin/python

echo "=============================="
echo "OASIS preprocessing started"
echo "Time: $(date)"
echo "=============================="

$PYTHON -u scripts/preprocess_oasis.py
$PYTHON -u scripts/preprocess_nii_to_npy_oasis.py

echo "=============================="
echo "Done at $(date)"
echo "=============================="
