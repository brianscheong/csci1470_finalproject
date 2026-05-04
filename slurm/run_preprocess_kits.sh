#!/bin/bash

#SBATCH -J preprocess_kits
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o preprocess.out
#SBATCH -e preprocess.err

# =========================================
# this is a shell script to run the preprocess_kits.py script on the cluster. It uses SLURM for job scheduling.
# You can submit this script using "sbatch run_preprocess_kits.sh" from the terminal. It allows me to run it without maintaining an active ssh connection to the cluster
# for the full several hours that it takes.
# terminal output will be saved to preprocess.out and preprocess.err files. You can check these files to monitor progress and debug as needed.
# =========================================


/oscar/rt/9.6/25/x86_64_v3/miniforge3-25.3.0-3-a6hhdjzejtacz63sugjqnvgosfqz63ul/bin/python -u scripts/preprocess_kits.py # conda wasn't working in the script below so this is the brute force 
# conda wasn't working in the script below so this is the brute force way
# the -u tag means unbuffered output, so that print statements are flushed out to preprocess/out immediately instead of needing to wait for the buffer to fill between writes. CS300!!!.



# cd /oscar/home/bcheong/science/voxelmorph_project
# module load python
# source ~/.bashrc
# conda activate voxelmorph
# python scripts/preprocess_kits.py