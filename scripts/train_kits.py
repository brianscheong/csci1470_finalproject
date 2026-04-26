import random
import numpy as np
import tensorflow as tf
import voxelmorph as vxm
import glob
import os
import json
# import nibabel as nib   # no longer needed


# =========================================
# This script trains a VoxelMorph model on the KiTS23 dataset. It uses a random pair generator, so it does not have a fixed epoch dataset pass.
# It assumes data is already preprocessed (Run preprocess_kits.py first to generate the preprocessed data). It saves the model weights at the end.
# =========================================

# ------- hyperparameters -------
lr = 1e-4 # PAPER
steps_per_epoch = 500
max_epochs = 50
ncc_weight = 1.0
grad_weight = 1.0 # chatGPT suggested 0.01, but papers suggests 1 is optimal if using NCC loss (fig 7)
int_steps = 7 # num steps of integrating velocity field to getr deformation field
int_downsize = 2 # downsample velocity field before integration for time/memory
val_steps = 50


# ------- load data -------
def load_case(case_dir):
    img = np.load(os.path.join(case_dir, "imaging.npy")).astype(np.float32)
    seg = np.load(os.path.join(case_dir, "segmentation.npy"))
    return img, seg

data_dir = "/oscar/scratch/bcheong/csci1470_data/kits23/dataset_preproc_npy/"
case_dirs = sorted(glob.glob(os.path.join(data_dir, "case_*")))
case_ids = [os.path.basename(c) for c in case_dirs]

print(f"Found {len(case_dirs)} cases")

# ------- construct train, test, validation sets (FROM JSON) -------
split_path = "/oscar/home/bcheong/science/csci1470_finalproject/kits_split.json"

with open(split_path, "r") as f:
    split = json.load(f)

train_ids = set(split["train"])
val_ids   = set(split["val"])
test_ids  = set(split["test"])

id_to_case = {cid: cdir for cid, cdir in zip(case_ids, case_dirs)}

train_cases = [id_to_case[cid] for cid in split["train"] if cid in id_to_case]
val_cases   = [id_to_case[cid] for cid in split["val"]   if cid in id_to_case]
test_cases  = [id_to_case[cid] for cid in split["test"]  if cid in id_to_case]

print(f"Train: {len(train_cases)}, Val: {len(val_cases)}, Test: {len(test_cases)}")
print(f"Expected: {len(train_ids)}, {len(val_ids)}, {len(test_ids)}")
print(f"Loaded:   {len(train_cases)}, {len(val_cases)}, {len(test_cases)}")


# ------- Random pair generator for training -------
zero_phi = np.zeros((1,160,192,224,3), dtype=np.float32) # template/allocation moved outside to avoid repeating

def preprocess(vol):
    # assumes dataset_preproc already resized to (160,192,224)
    return vol.astype(np.float32)

def train_generator():
    while True:
        i, j = random.sample(range(len(train_cases)), 2) # Randomly samples from the set of n(n-1) possible pairs

        moving_img, _ = load_case(train_cases[i])
        fixed_img,  _ = load_case(train_cases[j])

        moving = preprocess(moving_img)[np.newaxis, ..., np.newaxis]
        fixed  = preprocess(fixed_img)[np.newaxis, ..., np.newaxis]

        yield [moving, fixed], [fixed, zero_phi] # (inputs, targets)

def val_generator(): # same idea as train generator
    while True:
        i, j = random.sample(range(len(val_cases)), 2)

        moving_img, _ = load_case(val_cases[i])
        fixed_img,  _ = load_case(val_cases[j])

        moving = preprocess(moving_img)[np.newaxis, ..., np.newaxis]
        fixed  = preprocess(fixed_img)[np.newaxis, ..., np.newaxis]

        yield [moving, fixed], [fixed, zero_phi]


# ------- setup model to train -------
model = vxm.networks.VxmDense( # initialize weights (Keras default. Basically just random?)
    inshape=(160,192,224),
    src_feats=1, # num channels moving
    trg_feats=1, # num channels fixed
    int_steps=int_steps, # steps of integration (? ChatGPT)
    int_downsize=int_downsize # downsampling factor for integration.
)

losses = [
    vxm.losses.NCC().loss, # similarity loss [-1,1]
    vxm.losses.Grad('l2').loss # regularity loss
]

loss_weights = [ncc_weight, grad_weight] # regularization hyperparameters

model.compile( # like in Beras (CSCI 1470)
    optimizer=tf.keras.optimizers.Adam(lr), # PAPER
    loss=losses,
    loss_weights=loss_weights
)


# ------- train  model --------
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "weights/kits_vxm_best.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    min_delta=0,
    restore_best_weights=True
)

model.fit(
    train_generator(),
    steps_per_epoch=steps_per_epoch,
    epochs=max_epochs,
    validation_data=val_generator(),
    validation_steps=val_steps,
    callbacks=[early_stop, checkpoint]
)


# --------- save trained model weights ----------
print("\n Training successfully completed!")
os.makedirs("weights", exist_ok=True)
model.save("weights/kits_vxm_4_20.h5")


# --------- Report hyperparameters ----------
print("\n----- TRAINING SUMMARY -----")

print(f"Dataset size: {len(case_dirs)}")
print(f"Train volumes: {len(train_cases)}")
print(f"Validation volumes: {len(val_cases)}")
print(f"Validation steps: {val_steps}")

print("\nHyperparameters:")
print(f"Learning rate: {lr}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Max epochs: {max_epochs}")

print(f"NCC weight: {ncc_weight}")
print(f"Gradient regularization weight: {grad_weight}")

print(f"Integration steps: {int_steps}")
print(f"Integration downsize: {int_downsize}")

print(f"Validation steps: {val_steps}")