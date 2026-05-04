import os
import glob
import json
import random

# =========================================
# Generates oasis_split.json with an 80/10/10 train/val/test split
# over all preprocessed OASIS subjects.
# Run once before training.
# =========================================

data_dir = "/oscar/scratch/bcheong/csci1470_data/oasis/dataset_preproc_npy"
out_path = "/oscar/home/bcheong/science/csci1470_finalproject/oasis_split.json"

random.seed(42)

subj_dirs = sorted(glob.glob(os.path.join(data_dir, "OAS1_*")))
subj_ids  = [os.path.basename(s) for s in subj_dirs]

print(f"Total subjects: {len(subj_ids)}")

random.shuffle(subj_ids)

n       = len(subj_ids)
n_train = int(0.80 * n)
n_val   = int(0.10 * n)

train = subj_ids[:n_train]
val   = subj_ids[n_train:n_train + n_val]
test  = subj_ids[n_train + n_val:]

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

split = {"train": train, "val": val, "test": test}

with open(out_path, "w") as f:
    json.dump(split, f, indent=2)

print(f"Saved split to {out_path}")
