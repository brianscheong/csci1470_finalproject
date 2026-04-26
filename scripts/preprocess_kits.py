import os
import glob
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
import SimpleITK as sitk

# =========================================
# This script preprocesses the KiTS23 dataset for training VoxelMorph.
# Just as the original paper did, we will resample to 1mm isotropic, normalize intensity, and affine-align all cases to a common case (aka atlas).
# =========================================

# --------- CONFIG ---------
root_dir = "/oscar/scratch/bcheong/csci1470_data/kits23/dataset/"
out_root = "/oscar/scratch/bcheong/csci1470_data/kits23/dataset_preproc"

os.makedirs(out_root, exist_ok=True) # make output directory if it doesn't exist

target_spacing = (1,1,1) # VoxelMorph paper resamples to 1mm isotropic
target_shape = (160,192,224)

atlas_case = "case_00000" # needed for affine alignment. We will align all other cases to this one.

# --------- Helper Functions---------
def center_crop_or_pad(vol, shape): # to enforce same shape after anatomy-aware cropping
    out = np.zeros(shape, dtype=vol.dtype)
    min_shape = np.minimum(vol.shape, shape) # overlap of volume and target shape
    start_src = [(s - m)//2 for s,m in zip(vol.shape, min_shape)]
    start_dst = [(s - m)//2 for s,m in zip(shape, min_shape)]
    slices_src = tuple(slice(a,a+m) for a,m in zip(start_src,min_shape))
    slices_dst = tuple(slice(a,a+m) for a,m in zip(start_dst,min_shape))
    out[slices_dst] = vol[slices_src] # place centered, min_shaped-sized extraction from volume into out.  
    return out

def crop_to_mask(vol, seg, margin=20):
    mask = (seg > 0)
    coords = np.where(mask)

    if coords[0].size == 0:
        raise ValueError("Empty segmentation")

    zmin, ymin, xmin = np.min(coords, axis=1)
    zmax, ymax, xmax = np.max(coords, axis=1)

    zmin = max(0, zmin - margin)
    ymin = max(0, ymin - margin)
    xmin = max(0, xmin - margin)

    zmax = min(vol.shape[0], zmax + margin)
    ymax = min(vol.shape[1], ymax + margin)
    xmax = min(vol.shape[2], xmax + margin)

    vol_crop = vol[zmin:zmax, ymin:ymax, xmin:xmax]
    seg_crop = seg[zmin:zmax, ymin:ymax, xmin:xmax]

    return vol_crop, seg_crop

# affine alignment helpers
def to_sitk(img_np, is_label=False):
    img = sitk.GetImageFromArray(img_np)
    img.SetSpacing((1.0,1.0,1.0))
    return sitk.Cast(img, sitk.sitkUInt8 if is_label else sitk.sitkFloat32)

def affine_align(moving_np, fixed_np):
    moving = to_sitk(moving_np)
    fixed  = to_sitk(fixed_np)

    transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2)

    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    reg.SetInitialTransform(transform, inPlace=False)

    final_transform = reg.Execute(fixed, moving)

    aligned = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID()
    )

    return sitk.GetArrayFromImage(aligned), final_transform

def apply_transform_label(label_np, fixed_np, transform):
    moving = to_sitk(label_np, is_label=True)
    fixed  = to_sitk(fixed_np)

    aligned = sitk.Resample(
        moving,
        fixed,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        moving.GetPixelID()
    )

    return sitk.GetArrayFromImage(aligned)

# -------- LOAD ATLAS --------
atlas_img_path = os.path.join(root_dir, atlas_case, "imaging.nii.gz")
atlas_lab_path = os.path.join(root_dir, atlas_case, "segmentation.nii.gz")

# load NIfTI
atlas_img = nib.load(atlas_img_path)
atlas_lab = nib.load(atlas_lab_path)

# resample to 1mm (same as data)
atlas_img = resample_to_output(atlas_img, voxel_sizes=target_spacing, order=1) # order 1 is linear interpolation to get values for new spacing
atlas_lab = resample_to_output(atlas_lab, voxel_sizes=target_spacing, order=0) # order 0 is nearest neighbor interpolation. It preserves discrete labels so we don't get class 1.5. 
# order 2 is even smoother than order 1 (linear interpolation).

# convert to numpy
atlas_np = atlas_img.get_fdata()
atlas_seg = atlas_lab.get_fdata()

# crop using segmentation (shared function)
atlas_np, atlas_seg = crop_to_mask(atlas_np, atlas_seg, margin=20)

# normalize intensities (same as data)
atlas_np = np.clip(atlas_np, -200, 300)
atlas_np = (atlas_np + 200) / 500.0

# --------- Process Data ---------
cases = sorted(glob.glob(os.path.join(root_dir, "case_*"))) # list of all cases in the dataset

# keep only case_00563 → case_00588
cases = [c for c in cases if 565 <= int(os.path.basename(c).split('_')[1]) <= 588]

for i, c in enumerate(cases):
    case_id = os.path.basename(c)
    print(f"[{i+1}/{len(cases)}] {case_id}")

    img = nib.load(os.path.join(c, "imaging.nii.gz"))
    lab = nib.load(os.path.join(c, "segmentation.nii.gz"))

    img = resample_to_output(img, voxel_sizes=target_spacing, order=1)
    lab = resample_to_output(lab, voxel_sizes=target_spacing, order=0)

    # convert to np array
    vol = img.get_fdata()
    seg = lab.get_fdata() 

    # crop using segmentation
    vol, seg = crop_to_mask(vol, seg, margin=20)
    
    # normalize
    vol = np.clip(vol, -200, 300)
    vol = (vol + 200) / 500.0

    # affine alignment to atlas
    if case_id == atlas_case:
        vol_aligned = vol
        seg_aligned = seg 
    else:
        vol_aligned, T = affine_align(vol, atlas_np) # aligned volume and the optimized affine transformation from moving to fixed. We will use T to also align the segmentation (next line)
        seg_aligned = apply_transform_label(seg, atlas_np, T)

    # final reshape to target shape (with center crop or pad) 
    vol_final = center_crop_or_pad(vol_aligned, target_shape).astype(np.float32)
    seg_final = center_crop_or_pad(seg_aligned, target_shape).astype(np.uint8)

    # --- saving preprocessed case ---
    case_out_dir = os.path.join(out_root, case_id)
    os.makedirs(case_out_dir, exist_ok=True) # make case folder in dataset_preproc

    # convert numpy → NIfTI
    img_nifti = nib.Nifti1Image(vol_final, affine=np.eye(4))
    seg_nifti = nib.Nifti1Image(seg_final, affine=np.eye(4))

    # save
    nib.save(img_nifti, os.path.join(case_out_dir, "imaging.nii.gz"))
    nib.save(seg_nifti, os.path.join(case_out_dir, "segmentation.nii.gz"))    
print("Done.")