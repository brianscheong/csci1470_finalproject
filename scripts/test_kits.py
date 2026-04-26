#!/usr/bin/env python
import os
import glob
import numpy as np
import voxelmorph as vxm
import tensorflow as tf
import json

# =========================================
# This script computes the average Dice score of warped kidney segmentations (via random pairs sampled from test set)), using any VoxelMorph model (customize in LOAD MODEL).

# IMPORTANT: LOAD MODEL and SAVE RESULTS sections should be configured
# =========================================

# ---------------- CONFIG ----------------
data_dir = "/oscar/scratch/bcheong/csci1470_data/kits23/dataset_preproc_npy"
target_shape = (160, 192, 224)
np.random.seed(0)

# -------- LOAD MODEL (configure as needed) --------
# model_path = "weights/vxm_dense_brain_T1_3D_mse.h5" # pretrained on composite brain MRI set (see Vxm paper). This script is conducting a negative control.
model = vxm.networks.VxmDense( # random weights. true negative control.
    inshape=(160, 192, 224),
    src_feats=1,
    trg_feats=1,
    unet_half_res=True,
    int_steps=7,
    int_downsize=2
)
model_path = "N/A"
# model_path = "weights/kits_vxm_4_20.h5" # trained on KiTS but with improved training scheme--more images per epoch and validation to avoid overfitting (early stopped at ~25/50 epochs). 
# model = vxm.networks.VxmDense.load(model_path)

# -------- FUNCTIONS --------
def load_vol(case_id, is_label=False):
    case_dir = os.path.join(data_dir, case_id)

    if is_label:
        vol = np.load(os.path.join(case_dir, "segmentation.npy"))
    else:
        vol = np.load(os.path.join(case_dir, "imaging.npy")).astype(np.float32)

    vol = vol[np.newaxis, ..., np.newaxis]
    return vol

def dice(seg1, seg2):
    seg1 = (seg1 > 0)
    seg2 = (seg2 > 0)
    if seg1.sum() + seg2.sum() == 0:
        return 1.0
    return 2 * np.logical_and(seg1, seg2).sum() / (seg1.sum() + seg2.sum())

# -------- build case list --------
case_dirs = sorted(glob.glob(os.path.join(data_dir, "case_*")))
case_ids = [os.path.basename(c) for c in case_dirs]

# -------- load split from JSON --------
split_path = "/oscar/home/bcheong/science/csci1470_finalproject/kits_split.json"

with open(split_path, "r") as f:
    split = json.load(f)

train_cases = [cid for cid in split["train"] if cid in case_ids]
val_cases   = [cid for cid in split["val"]   if cid in case_ids]
test_cases  = [cid for cid in split["test"]  if cid in case_ids]

print(f"Train: {len(train_cases)}, Val: {len(val_cases)}, Test: {len(test_cases)}")
print(load_vol(test_cases[0]).shape)

# -------- label warping layer --------
label_transform = vxm.networks.Transform(
    target_shape, nb_feats=1, interp_method="nearest" # nearest neighbor interpolation to preserve discrete labels. Default is linear which would give us non-integer values for the warped segmentation, which don't make sense as class labels..
)

# -------- MAIN LOOP  --------
num_pairs = 100 # or 200 if you want more stable stats

dice_scores = []

for k in range(num_pairs):
    case_i, case_j = np.random.choice(test_cases, size=2, replace=False) # sample random pair from test set without replacement

    moving_img = load_vol(case_i, is_label=False)
    fixed_img  = load_vol(case_j, is_label=False)

    moving_lab = load_vol(case_i, is_label=True)
    fixed_lab  = load_vol(case_j, is_label=True)

    d = dice(moving_lab.squeeze(), fixed_lab.squeeze()) # to test no-warp case (negative control)

    dice_scores.append(d)

    print(
        f"Pair {k+1}: {case_i} -> {case_j} | "
        f"Dice = {d:.4f} | "
        f"Running mean = {np.mean(dice_scores):.4f}"
    )

print("------------------------------------------------")
print(f"Mean Dice: {np.mean(dice_scores):.4f}")
print(f"Std Dice:  {np.std(dice_scores):.4f}")

# -------- save results to text file (CONFIGURE AS NEEDED)--------
os.makedirs("results", exist_ok=True)
out_path = os.path.join("results", "dice_results_kits_4_20_no_warp.txt")
with open(out_path, "w") as f:
    f.write(f"Model: {model_path}\n")
    f.write(f"Num subjects: {len(dice_scores)}\n")
    f.write(f"Mean Dice: {np.mean(dice_scores):.6f}\n")
    f.write(f"Std Dice: {np.std(dice_scores):.6f}\n")