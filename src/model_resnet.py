#!/usr/bin/env python3
import os
import argparse
import datetime
import json
import yaml
import numpy as np
import tensorflow as tf
from glob import glob
import math
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------------------------------------------------------
# 1) CLI: short tag for this run
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--tag", type=str, default="v1",
    help="short tag to describe this run (e.g. 'resnetv2_aug')"
)
args = parser.parse_args()

# -----------------------------------------------------------------------------
# 2) Paths & dirs
# -----------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CFG_PATH     = os.path.join(SCRIPT_DIR, "config.yaml")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "interim")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
OUT_DIR      = os.path.join(PROJECT_ROOT, "outputs", "metrics")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR,   exist_ok=True)

# -----------------------------------------------------------------------------
# 3) Load config
# -----------------------------------------------------------------------------
with open(CFG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

SR      = cfg["audio"]["sample_rate"]
CHUNK   = cfg["audio"]["chunk_duration"]
N_MELS  = cfg.get("n_mels", 128)
FFT     = cfg.get("fft_size", 1024)
HOP     = cfg.get("hop_length", 512)
BATCH   = cfg.get("batch_size", 32)
EPOCHS  = cfg.get("epochs", 20)
species = cfg["species"]

# -----------------------------------------------------------------------------
# 4) Gather files & labels
# -----------------------------------------------------------------------------
wavs = sorted(glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True))
if not wavs:
    raise RuntimeError(f"No WAVs found in {DATA_DIR}")

label_map = {sp: i for i, sp in enumerate(species)}
labels = []
for p in wavs:
    name = os.path.basename(p).rsplit("_", 2)[0].replace("_", " ")
    if name not in label_map:
        raise KeyError(f"Unknown species '{name}' in {p}")
    labels.append(label_map[name])

# 80/20 split
idxs = np.arange(len(wavs))
np.random.seed(42); np.random.shuffle(idxs)
cut = int(0.8 * len(idxs))
tr, va = idxs[:cut], idxs[cut:]
train_files = [wavs[i] for i in tr]; train_labels = [labels[i] for i in tr]
val_files   = [wavs[i] for i in va]; val_labels   = [labels[i] for i in va]

steps_per_epoch  = math.ceil(len(train_files) / BATCH)
validation_steps = math.ceil(len(val_files)   / BATCH)

# -----------------------------------------------------------------------------
# 5) SpecAugment decode→log-Mel w/ time & freq masks
# -----------------------------------------------------------------------------
def decode_and_mel(path, label):
    # load
    raw = tf.io.read_file(path)
    wav, _ = tf.audio.decode_wav(raw,
        desired_channels=1,
        desired_samples=SR * CHUNK
    )
    wav = tf.squeeze(wav, -1)
    # STFT → Mel
    spec = tf.abs(tf.signal.stft(wav, FFT, HOP))
    mel_wt = tf.signal.linear_to_mel_weight_matrix(N_MELS, spec.shape[-1], SR)
    mel = tf.tensordot(spec, mel_wt, 1)
    mel = tf.math.log(mel + 1e-6)           # [time, n_mels]
    # SpecAugment
    T = tf.shape(mel)[0]; F = tf.shape(mel)[1]
    # time mask
    t0 = tf.random.uniform([], 0, T//5, dtype=tf.int32)
    mel = tf.concat([mel[:t0], tf.zeros([T//5, F]), mel[t0+T//5:]], axis=0)
    # freq mask
    f0 = tf.random.uniform([], 0, F//5, dtype=tf.int32)
    mel = tf.concat([mel[:, :f0], tf.zeros([T, F//5]), mel[:, f0+F//5:]], axis=1)
    mel = tf.expand_dims(tf.transpose(mel), -1)  # [n_mels, time, 1]
    return mel, label

def make_ds(files, labs):
    ds = tf.data.Dataset.from_tensor_slices((files, labs))
    ds = ds.map(decode_and_mel, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(train_files, train_labels).repeat()
val_ds   = make_ds(val_files,   val_labels).repeat()

# -----------------------------------------------------------------------------
# 6) Residual block
# -----------------------------------------------------------------------------
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization,
    Activation, Add, MaxPool2D, Dropout, GlobalAveragePooling2D, Dense, Input
)
def residual_block(x, filters, kernel_size=3):
    shortcut = x
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    if int(shortcut.shape[-1]) != filters:
        shortcut = Conv2D(filters, 1, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([shortcut, x])
    return Activation('relu')(x)

# -----------------------------------------------------------------------------
# 7) Build a 4-block ResNet
# -----------------------------------------------------------------------------
inp = Input((N_MELS, None, 1))
x = residual_block(inp,  32)
x = MaxPool2D()(x); x = Dropout(0.2)(x)
x = residual_block(x,   64)
x = MaxPool2D()(x); x = Dropout(0.2)(x)
x = residual_block(x,  128)
x = MaxPool2D()(x); x = Dropout(0.2)(x)
x = residual_block(x,  256)
x = MaxPool2D()(x); x = Dropout(0.2)(x)

x = GlobalAveragePooling2D()(x)
x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)
out = Dense(len(species), activation='softmax')(x)

model = tf.keras.Model(inp, out)

# -----------------------------------------------------------------------------
# 8) Cosine-decay Adam
# -----------------------------------------------------------------------------
decay_steps = EPOCHS * steps_per_epoch
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(1e-3, decay_steps)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=opt,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# -----------------------------------------------------------------------------
# 9) Callbacks
# -----------------------------------------------------------------------------
ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
chk = f"bird_resnet4_{args.tag}_{ts}.keras"
ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(MODEL_DIR, chk),
    save_best_only=True,
    monitor="val_accuracy"
)
es_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

# -----------------------------------------------------------------------------
# 10) Train
# -----------------------------------------------------------------------------
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    callbacks=[ckpt_cb, es_cb],
    verbose=1
)
print(f"\n Done! Best model: models/{chk}")

# -----------------------------------------------------------------------------
# 11) Save history & metrics & conf matrix
# -----------------------------------------------------------------------------
# history
with open(os.path.join(OUT_DIR, f"history_{args.tag}_{ts}.json"), "w") as f:
    json.dump(history.history, f)
# metrics
eval_res = model.evaluate(val_ds, steps=validation_steps, verbose=0)
with open(os.path.join(OUT_DIR, f"metrics_{args.tag}_{ts}.json"), "w") as f:
    json.dump(dict(zip(model.metrics_names, eval_res)), f)
# confusion + report
preds, trues = [], []
for fp, lbl in zip(val_files, val_labels):
    m, _ = decode_and_mel(fp, lbl)
    p = model.predict(tf.expand_dims(m,0), verbose=0)[0]
    preds.append(np.argmax(p))
    trues.append(lbl)

# save confusion matrix
cm = confusion_matrix(trues, preds)
np.savetxt(os.path.join(OUT_DIR, f"cm_{args.tag}_{ts}.csv"), cm, delimiter=",", fmt="%d")

unique_labels = sorted(set(trues))
unique_names  = [species[i] for i in unique_labels]

# write classification report
report = classification_report(
    trues,
    preds,
    labels=unique_labels,
    target_names=unique_names
)
with open(os.path.join(OUT_DIR, f"report_{args.tag}_{ts}.txt"), "w") as f:
    f.write(report)
print(" Classification report saved")

