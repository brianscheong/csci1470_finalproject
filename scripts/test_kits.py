#!/usr/bin/env python
import os
import json
import glob
import argparse
import numpy as np
import voxelmorph as vxm
import tensorflow as tf

# =========================================
# Evaluates a trained VoxelMorph model on the KiTS23 test set.
# Samples random pairs, warps the moving segmentation using the predicted
# deformation field, and reports mean Dice score.
#
# Usage:
#   python test_kits.py --model weights/kits_vxm_best.h5 \
#                       --out results/dice_kits_trained.txt \
#                       --num_pairs 100
#
# For the no-warp baseline, pass --model none
# For the classical baseline, pass --model classical
# =========================================

parser = argparse.ArgumentParser()
parser.add_argument("--model",     default="weights/kits_vxm_best.h5",
                    help="Path to .h5 model, 'none' for no-warp, or 'classical' for ANTs")
parser.add_argument("--num_pairs", type=int, default=100,
                    help="Number of random test pairs to evaluate")
parser.add_argument("--seed",      type=int, default=0)
args = parser.parse_args()

# ------- Paths -------
DATA_DIR   = "/oscar/scratch/bcheong/csci1470_data/kits23/dataset_preproc_npy"
SPLIT_PATH = "/oscar/home/bcheong/science/csci1470_finalproject/kits_split.json"

target_shape = (160, 192, 224)
np.random.seed(args.seed)

# ------- Load split -------
with open(SPLIT_PATH) as f:
    split = json.load(f)

all_ids = {os.path.basename(d)
           for d in glob.glob(os.path.join(DATA_DIR, "case_*"))}

test_cases = [cid for cid in split["test"] if cid in all_ids]
print(f"Test cases: {len(test_cases)}")

# ------- Load model -------
mode = args.model.lower()
if mode == "none":
    print("Mode: no-warp baseline (negative control)")
    model = None
elif mode == "classical":
    print("Mode: classical registration (SimpleITK affine)")
    model = "classical"
else:
    print(f"Mode: VoxelMorph — loading {args.model}")
    model = vxm.networks.VxmDense.load(args.model)

# ------- Label warping layer -------
label_transform = vxm.networks.Transform(
    target_shape,
    nb_feats=1,
    interp_method="nearest",
)

# ------- Helper functions -------
def load_vol(cid, is_label=False):
    case_dir = os.path.join(DATA_DIR, cid)
    fname    = "segmentation.npy" if is_label else "imaging.npy"
    vol      = np.load(os.path.join(case_dir, fname))
    dtype    = np.float32
    vol      = vol.astype(dtype)
    return vol[np.newaxis, ..., np.newaxis]  # (1, D, H, W, 1)

def dice(seg1, seg2):
    """Binary Dice: any nonzero label vs background."""
    a = (seg1 > 0).astype(bool)
    b = (seg2 > 0).astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return 2.0 * np.logical_and(a, b).sum() / denom

def classical_register_and_dice(moving_img_np, fixed_img_np,
                                 moving_seg_np, fixed_seg_np):
    """
    Run SimpleITK affine registration between a moving/fixed pair
    and return the Dice score of the warped moving segmentation.
    """
    import SimpleITK as sitk

    def to_sitk_img(arr):
        img = sitk.GetImageFromArray(arr.astype(np.float32))
        img.SetSpacing((1.0, 1.0, 1.0))
        return img

    def to_sitk_seg(arr):
        img = sitk.GetImageFromArray(arr.astype(np.uint8))
        img.SetSpacing((1.0, 1.0, 1.0))
        return img

    moving_sitk = to_sitk_img(moving_img_np)
    fixed_sitk  = to_sitk_img(fixed_img_np)

    # Initialise affine transform at geometry centre
    tx = sitk.CenteredTransformInitializer(
        fixed_sitk, moving_sitk,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(tx, inPlace=False)

    final_tx = reg.Execute(fixed_sitk, moving_sitk)

    # Warp segmentation with nearest-neighbour interpolation
    warped_seg_sitk = sitk.Resample(
        to_sitk_seg(moving_seg_np),
        fixed_sitk,
        final_tx,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    warped_seg_np = sitk.GetArrayFromImage(warped_seg_sitk).astype(np.float32)

    return dice(warped_seg_np, fixed_seg_np)

# ------- Evaluation loop -------
import time
dice_scores = []
times       = []

for k in range(args.num_pairs):
    cid_m, cid_f = np.random.choice(test_cases, size=2, replace=False)

    moving_img = load_vol(cid_m, is_label=False)
    fixed_img  = load_vol(cid_f, is_label=False)
    moving_seg = load_vol(cid_m, is_label=True)
    fixed_seg  = load_vol(cid_f, is_label=True)

    t0 = time.time()

    if model is None:
        # No-warp baseline
        warped_seg = moving_seg
        d = dice(warped_seg.squeeze(), fixed_seg.squeeze())

    elif model == "classical":
        d = classical_register_and_dice(
            moving_img.squeeze(), fixed_img.squeeze(),
            moving_seg.squeeze(), fixed_seg.squeeze(),
        )

    else:
        # VoxelMorph
        _, warp_field = model.predict([moving_img, fixed_img], verbose=0)
        warped_seg    = label_transform.predict([moving_seg, warp_field], verbose=0)
        d = dice(warped_seg.squeeze(), fixed_seg.squeeze())

    elapsed = time.time() - t0
    dice_scores.append(d)
    times.append(elapsed)

    print(
        f"Pair {k+1:>3}: {cid_m} → {cid_f} | "
        f"Dice = {d:.4f} | time = {elapsed:.1f}s | "
        f"Running mean = {np.mean(dice_scores):.4f}"
    )

mean_dice  = np.mean(dice_scores)
std_dice   = np.std(dice_scores)
total_time = np.sum(times)

print("------------------------------------------------")
print(f"Mean Dice : {mean_dice:.4f}")
print(f"Std  Dice : {std_dice:.4f}")
print(f"Total time: {total_time:.1f}s over {args.num_pairs} pairs")


