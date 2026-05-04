import os
import random
import json
import glob
import numpy as np
import tensorflow as tf
import voxelmorph as vxm

# =========================================
# Trains a VoxelMorph model on OASIS brain MRI data.
# Architecture and training protocol follow the original VoxelMorph paper.
#
# Experiments controlled by this script:
#   Experiment 1 (claim verification)  — train with CC loss, lambda=1.0, full dataset
#   Experiment 2 (training set size)   — use train_size argument (10, 100, or "full")
#   Experiment 3 (smoothness ablation) — vary GRAD_WEIGHT
#   Experiment 4 (loss ablation)       — vary LOSS_TYPE
#
# For ablation sweeps, see run_ablation_lambda.sh and run_ablation_loss.sh which
# call this script with different environment-variable overrides.
# =========================================

# ------- Hyperparameters (override via env vars for ablation sweeps) -------
LR              = float(os.environ.get("LR",           "1e-4"))
STEPS_PER_EPOCH = int(os.environ.get("STEPS_PER_EPOCH","500"))
MAX_EPOCHS      = int(os.environ.get("MAX_EPOCHS",     "150"))
VAL_STEPS       = int(os.environ.get("VAL_STEPS",      "50"))
INT_STEPS       = int(os.environ.get("INT_STEPS",      "7"))
INT_DOWNSIZE    = int(os.environ.get("INT_DOWNSIZE",   "2"))

# Experiment 3: smoothness regularization weight
GRAD_WEIGHT = float(os.environ.get("GRAD_WEIGHT", "1.0"))

# Experiment 4: image similarity loss  — "ncc", "mse", or "ssim"
LOSS_TYPE = os.environ.get("LOSS_TYPE", "ncc").lower()

# Experiment 2: training set size limit (int or "full")
TRAIN_SIZE = os.environ.get("TRAIN_SIZE", "full")

# Output name suffix so ablation runs don't overwrite each other
RUN_TAG = os.environ.get("RUN_TAG", "default")

# ------- Paths -------
DATA_DIR   = "/oscar/scratch/bcheong/csci1470_data/oasis/dataset_preproc_npy"
SPLIT_PATH = "/oscar/home/bcheong/science/csci1470_finalproject/oasis_split.json"

# ------- Load split -------
with open(SPLIT_PATH) as f:
    split = json.load(f)

all_case_dirs = {os.path.basename(d): d
                 for d in sorted(glob.glob(os.path.join(DATA_DIR, "OAS1_*")))}

train_cases = [all_case_dirs[sid] for sid in split["train"] if sid in all_case_dirs]
val_cases   = [all_case_dirs[sid] for sid in split["val"]   if sid in all_case_dirs]

# Experiment 2: cap training set size
if TRAIN_SIZE != "full":
    n = int(TRAIN_SIZE)
    random.seed(42)
    train_cases = random.sample(train_cases, min(n, len(train_cases)))

print(f"Train: {len(train_cases)}, Val: {len(val_cases)}")
print(f"Loss: {LOSS_TYPE}, grad_weight: {GRAD_WEIGHT}, tag: {RUN_TAG}")

# ------- Data loading -------
def load_case(case_dir):
    img = np.load(os.path.join(case_dir, "brain.npy")).astype(np.float32)
    return img  # shape (160,192,224)

zero_phi = np.zeros((1, 160, 192, 224, 3), dtype=np.float32)

def make_generator(cases):
    def _gen():
        while True:
            i, j = random.sample(range(len(cases)), 2)
            moving = load_case(cases[i])[np.newaxis, ..., np.newaxis]
            fixed  = load_case(cases[j])[np.newaxis, ..., np.newaxis]
            yield [moving, fixed], [fixed, zero_phi]
    return _gen

# ------- Build model -------
model = vxm.networks.VxmDense(
    inshape=(160, 192, 224),
    src_feats=1,
    trg_feats=1,
    int_steps=INT_STEPS,
    int_downsize=INT_DOWNSIZE,
    unet_half_res=True,
)

# ------- Select similarity loss (Experiment 4) -------
if LOSS_TYPE == "ncc":
    sim_loss = vxm.losses.NCC().loss
elif LOSS_TYPE == "mse":
    sim_loss = vxm.losses.MSE().loss
elif LOSS_TYPE == "ssim":
    # SSIM loss: 1 - mean SSIM over local windows.
    # voxelmorph does not ship SSIM natively; we implement a simple version.
    import tensorflow as tf

    def ssim_loss(y_true, y_pred):
        # Operate on 2-D slices along the last spatial axis to keep memory manageable.
        # y_true / y_pred shape: (batch, D, H, W, 1)
        # Average SSIM across the D dimension of axial slices.
        slices_true = tf.unstack(y_true[:, :, :, :, 0], axis=1)  # list of (B,H,W)
        slices_pred = tf.unstack(y_pred[:, :, :, :, 0], axis=1)
        ssim_vals = []
        for st, sp in zip(slices_true, slices_pred):
            st_exp = tf.expand_dims(st, -1)  # (B,H,W,1)
            sp_exp = tf.expand_dims(sp, -1)
            s = tf.image.ssim(st_exp, sp_exp, max_val=1.0)
            ssim_vals.append(s)
        mean_ssim = tf.reduce_mean(tf.stack(ssim_vals, axis=1), axis=1)
        return 1.0 - tf.reduce_mean(mean_ssim)

    sim_loss = ssim_loss
else:
    raise ValueError(f"Unknown LOSS_TYPE: {LOSS_TYPE}")

losses       = [sim_loss, vxm.losses.Grad("l2").loss]
loss_weights = [1.0, GRAD_WEIGHT]

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss=losses,
    loss_weights=loss_weights,
)

# ------- Callbacks -------
os.makedirs("weights", exist_ok=True)
best_path = f"weights/oasis_vxm_{RUN_TAG}_best.h5"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        best_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    ),
]

# ------- Train -------
model.fit(
    make_generator(train_cases)(),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=MAX_EPOCHS,
    validation_data=make_generator(val_cases)(),
    validation_steps=VAL_STEPS,
    callbacks=callbacks,
)

# ------- Save final weights -------
final_path = f"weights/oasis_vxm_{RUN_TAG}_final.h5"
model.save(final_path)

print("\n===== TRAINING COMPLETE =====")
print(f"Best weights : {best_path}")
print(f"Final weights: {final_path}")
print(f"Loss type    : {LOSS_TYPE}")
print(f"Grad weight  : {GRAD_WEIGHT}")
print(f"Train size   : {len(train_cases)}")
print(f"LR           : {LR}")
print(f"Steps/epoch  : {STEPS_PER_EPOCH}")
print(f"Max epochs   : {MAX_EPOCHS}")
