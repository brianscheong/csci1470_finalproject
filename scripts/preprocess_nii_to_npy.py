# =========================================
# This script is just part two of preprocesisng. It was just used to get the preprocessed .nii.gz files into .npy format for easier loading during training.
# =========================================

import os
import glob
import numpy as np
import nibabel as nib

in_dir  = "/oscar/scratch/bcheong/csci1470_data/kits23/dataset_preproc"
out_dir = "/oscar/scratch/bcheong/csci1470_data/kits23/dataset_preproc_npy"

case_dirs = sorted(glob.glob(os.path.join(in_dir, "case_*")))

for c in case_dirs:
    cid = os.path.basename(c)

    out_case_dir = os.path.join(out_dir, cid)
    os.makedirs(out_case_dir, exist_ok=True)

    # --- imaging ---
    img_path = os.path.join(c, "imaging.nii.gz")
    if os.path.exists(img_path):
        img = nib.load(img_path).get_fdata().astype(np.float32)
        np.save(os.path.join(out_case_dir, "imaging.npy"), img)

    # --- segmentation ---
    lab_path = os.path.join(c, "segmentation.nii.gz")
    if os.path.exists(lab_path):
        lab = nib.load(lab_path).get_fdata().astype(np.int16)
        np.save(os.path.join(out_case_dir, "segmentation.npy"), lab)

    print(f"Saved {cid}")