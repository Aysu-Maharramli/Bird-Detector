#!/usr/bin/env python3
import os
import argparse
import datetime
import yaml
import numpy as np
import tensorflow as tf
from glob import glob
import math

# -----------------------------------------------------------------------------
# 1) CLI: allow a short tag for checkpoint filenames
# -----------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument(
    "--tag",
    type=str,
    default="v1",
    help="short tag to describe this run (e.g. 'baseline')"
)
args = p.parse_args()

# -----------------------------------------------------------------------------
# 2) Paths: locate config and data directories
# -----------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CFG_PATH     = os.path.join(SCRIPT_DIR, "config.yaml")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "interim")

# -----------------------------------------------------------------------------
# 3) Load hyperparameters from config.yaml
# -----------------------------------------------------------------------------
cfg     = yaml.safe_load(open(CFG_PATH))
SR      = cfg["audio"]["sample_rate"]
CHUNK   = cfg["audio"]["chunk_duration"]
N_MELS  = cfg.get("n_mels", 128)
FFT     = cfg.get("fft_size", 1024)
HOP     = cfg.get("hop_length", 512)
BATCH   = cfg.get("batch_size", 32)
EPOCHS  = cfg.get("epochs", 20)

# -----------------------------------------------------------------------------
# 4) Gather .wav files and build label mappings
# -----------------------------------------------------------------------------
wav_paths = sorted(glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True))
if not wav_paths:
    raise RuntimeError(f"No .wav files found in {DATA_DIR}")

species   = cfg["species"]
label_map = {sp: i for i, sp in enumerate(species)}

all_labels = []
for p in wav_paths:
    # extracting species from filename
    name = os.path.basename(p).rsplit("_", 2)[0].replace("_", " ")
    if name not in label_map:
        raise KeyError(f"Species '{name}' not in config.species")
    all_labels.append(label_map[name])

# train/validation split
idxs = np.arange(len(wav_paths))
np.random.seed(42)
np.random.shuffle(idxs)
split      = int(0.8 * len(idxs))
train_idxs = idxs[:split]
val_idxs   = idxs[split:]

train_files  = [wav_paths[i] for i in train_idxs]
train_labels = [all_labels[i] for i in train_idxs]
val_files    = [wav_paths[i] for i in val_idxs]
val_labels   = [all_labels[i] for i in val_idxs]

# -----------------------------------------------------------------------------
# 5) Audio preprocessing function: WAV -> log-Mel
# -----------------------------------------------------------------------------
def decode_and_mel(path, label):
    raw = tf.io.read_file(path)
    wav, _ = tf.audio.decode_wav(raw,
        desired_channels=1,
        desired_samples=SR * CHUNK
    )
    wav = tf.squeeze(wav, -1)
    spec = tf.signal.stft(wav, frame_length=FFT, frame_step=HOP)
    spec = tf.abs(spec)

    mel_wt = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=spec.shape[-1],
        sample_rate=SR
    )
    mel = tf.tensordot(spec, mel_wt, 1)
    mel = tf.math.log(mel + 1e-6)
    mel = tf.transpose(mel)
    return tf.expand_dims(mel, -1), label

# -----------------------------------------------------------------------------
# 6) Build tf.data pipelines and compute steps
# -----------------------------------------------------------------------------
train_ds = (tf.data.Dataset
    .from_tensor_slices((
        tf.constant(train_files, dtype=tf.string),
        tf.constant(train_labels, dtype=tf.int32)
    ))
    .map(decode_and_mel, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (tf.data.Dataset
    .from_tensor_slices((
        tf.constant(val_files, dtype=tf.string),
        tf.constant(val_labels, dtype=tf.int32)
    ))
    .map(decode_and_mel, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

steps_per_epoch  = math.ceil(len(train_files) / BATCH)
validation_steps = math.ceil(len(val_files)   / BATCH)

# -----------------------------------------------------------------------------
# 7) Build the CNN model
# -----------------------------------------------------------------------------
inputs = tf.keras.layers.Input(shape=(N_MELS, None, 1))
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPool2D()(x)

x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPool2D()(x)

x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPool2D()(x)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
outputs = tf.keras.layers.Dense(len(species), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# -----------------------------------------------------------------------------
# 8) Checkpointing
# -----------------------------------------------------------------------------
os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
name = f"bird_cnn_{args.tag}_{ts}.keras"
ckpt = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(PROJECT_ROOT, "models", name),
    save_best_only=True,
    monitor="val_accuracy"
)

# -----------------------------------------------------------------------------
# 9) Train with verbose output
# -----------------------------------------------------------------------------
# repeat datasets for multiple epochs
train_ds_r = train_ds.repeat()
val_ds_r   = val_ds.repeat()

history = model.fit(
    train_ds_r,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds_r,
    validation_steps=validation_steps,
    callbacks=[ckpt],
    verbose=1
)

print(f"\n Training complete! Best model saved to models/{name}")
