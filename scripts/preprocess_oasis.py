import os
import glob
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output

# =========================================
# This script preprocesses the OASIS-1 dataset for training VoxelMorph.
# Following the original VoxelMorph paper:
#   - Resample to 1mm isotropic voxels
#   - Normalize intensity to [0, 1] by clipping and rescaling
#   - Center-crop / pad to a fixed volume shape
#
# OASIS-1 directory structure assumed:
#   <root_dir>/OAS1_XXXX_MR1/
#       mri/
#           brain.mgz           (FreeSurfer skull-stripped brain)
#           aparc+aseg.mgz      (FreeSurfer cortical/subcortical parcellation)
#
# Run this script before training. Output is saved as .nii.gz files,
# then convert to .npy using preprocess_nii_to_npy_oasis.py.
# =========================================

# --------- CONFIG ---------
root_dir  = "/oscar/scratch/bcheong/csci1470_data/oasis/disc1"   # adjust to your OASIS download path
out_root  = "/oscar/scratch/bcheong/csci1470_data/oasis/dataset_preproc"

os.makedirs(out_root, exist_ok=True)

target_spacing = (1, 1, 1)       # 1mm isotropic, same as VoxelMorph paper
target_shape   = (160, 192, 224)  # match KiTS shape so the same model arch works

# --------- Helper Functions ---------
def center_crop_or_pad(vol, shape):
    """Crop or zero-pad vol symmetrically to reach target shape."""
    out = np.zeros(shape, dtype=vol.dtype)
    min_shape  = np.minimum(vol.shape, shape)
    start_src  = [(s - m) // 2 for s, m in zip(vol.shape, min_shape)]
    start_dst  = [(s - m) // 2 for s, m in zip(shape, min_shape)]
    slices_src = tuple(slice(a, a + m) for a, m in zip(start_src, min_shape))
    slices_dst = tuple(slice(a, a + m) for a, m in zip(start_dst, min_shape))
    out[slices_dst] = vol[slices_src]
    return out

def normalize_brain(vol):
    """Clip to 99th percentile then rescale to [0, 1]."""
    p99 = np.percentile(vol[vol > 0], 99)
    vol = np.clip(vol, 0, p99)
    vol = vol / (p99 + 1e-8)
    return vol.astype(np.float32)

# --------- Find all subjects ---------
# OASIS-1 disc layout: OAS1_XXXX_MR1/
subject_dirs = sorted(glob.glob(os.path.join(root_dir, "OAS1_*_MR1")))
print(f"Found {len(subject_dirs)} OASIS subjects")

skipped = []

for i, subj_dir in enumerate(subject_dirs):
    subj_id = os.path.basename(subj_dir)

    img_path = os.path.join(subj_dir, "mri", "brain.mgz")
    seg_path = os.path.join(subj_dir, "mri", "aparc+aseg.mgz")

    if not os.path.exists(img_path) or not os.path.exists(seg_path):
        print(f"  [{i+1}/{len(subject_dirs)}] SKIP {subj_id} — missing mri files")
        skipped.append(subj_id)
        continue

    print(f"[{i+1}/{len(subject_dirs)}] {subj_id}")

    # --- load ---
    img_nib = nib.load(img_path)
    seg_nib = nib.load(seg_path)

    # --- resample to 1mm isotropic ---
    img_nib = resample_to_output(img_nib, voxel_sizes=target_spacing, order=1)
    seg_nib = resample_to_output(seg_nib, voxel_sizes=target_spacing, order=0)

    vol = img_nib.get_fdata(dtype=np.float32)
    seg = seg_nib.get_fdata().astype(np.int32)

    # --- normalize ---
    vol = normalize_brain(vol)

    # --- center crop/pad ---
    vol_final = center_crop_or_pad(vol, target_shape).astype(np.float32)
    seg_final = center_crop_or_pad(seg, target_shape).astype(np.int32)

    # --- save ---
    out_dir = os.path.join(out_root, subj_id)
    os.makedirs(out_dir, exist_ok=True)

    nib.save(nib.Nifti1Image(vol_final, np.eye(4)),
             os.path.join(out_dir, "brain.nii.gz"))
    nib.save(nib.Nifti1Image(seg_final.astype(np.int32), np.eye(4)),
             os.path.join(out_dir, "aparc+aseg.nii.gz"))

print(f"\nDone. Skipped {len(skipped)} subjects: {skipped}")
