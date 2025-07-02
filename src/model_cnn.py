#!/usr/bin/env python3
import os, yaml, numpy as np, tensorflow as tf
from glob import glob

# 1) Config & hyperparams
cfg    = yaml.safe_load(open("src/config.yaml"))
SR, CHUNK = cfg["audio"]["sample_rate"], cfg["audio"]["chunk_duration"]
N_MELS, FFT, HOP = 128, 1024, 512
BATCH, EPOCHS = 64, 20

# 2) Label map
species   = cfg["species"]
label_map = {sp: i for i, sp in enumerate(species)}

# 3) File list + shuffle & split
wav_paths = glob("data/interim/**/*.wav", recursive=True)
labels    = [
    label_map[
      os.path.basename(p).rsplit("_", 2)[0].replace("_", " ")
    ]
    for p in wav_paths
]
idxs = np.arange(len(wav_paths))
np.random.seed(42)
np.random.shuffle(idxs)
split = int(0.8 * len(idxs))
train_idx, val_idx = idxs[:split], idxs[split:]
train_files = [wav_paths[i] for i in train_idx]
train_labels= [labels[i]    for i in train_idx]
val_files   = [wav_paths[i] for i in val_idx]
val_labels  = [labels[i]    for i in val_idx]

train_ds = tf.data.Dataset.from_tensor_slices(
    (tf.constant(train_files, dtype=tf.string),
     tf.constant(train_labels, dtype=tf.int32))
)
val_ds = tf.data.Dataset.from_tensor_slices(
    (tf.constant(val_files, dtype=tf.string),
     tf.constant(val_labels, dtype=tf.int32))
)

# 4) Decode→mel fn
def decode_and_mel(path, label):
    raw = tf.io.read_file(path)
    wav, _ = tf.audio.decode_wav(raw,
        desired_channels=1,
        desired_samples=SR*CHUNK
    )
    wav = tf.squeeze(wav, -1)
    spec = tf.signal.stft(wav, frame_length=FFT, frame_step=HOP)
    spec = tf.abs(spec)
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=spec.shape[-1],
        sample_rate=SR
    )
    mel = tf.tensordot(spec, mel_matrix, 1)
    mel = tf.math.log(mel + 1e-6)
    mel = tf.transpose(mel)           # [N_MELS, time]
    return tf.expand_dims(mel, -1), label  # [N_MELS, time, 1]

# 5) Build datasets
def make_ds(ds, shuffle=False):
    if shuffle:
        ds = ds.shuffle(10_000)
    return (ds
      .map(decode_and_mel, num_parallel_calls=tf.data.AUTOTUNE)
      .batch(BATCH)
      .prefetch(tf.data.AUTOTUNE)
    )

train_ds = make_ds(train_ds, shuffle=True).repeat()
val_ds   = make_ds(val_ds).repeat()

# 6) Model
inp = tf.keras.layers.Input((N_MELS, None, 1))
x   = tf.keras.layers.Conv2D(32, (3,3), activation="relu")(inp)
x   = tf.keras.layers.MaxPool2D((2,2))(x)
x   = tf.keras.layers.Conv2D(64, (3,3), activation="relu")(x)
x   = tf.keras.layers.MaxPool2D((2,2))(x)
x   = tf.keras.layers.GlobalAveragePooling2D()(x)
x   = tf.keras.layers.Dense(128, activation="relu")(x)
out = tf.keras.layers.Dense(len(species), activation="softmax")(x)
model = tf.keras.Model(inp, out)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# 7) Train + checkpoint best
os.makedirs("models", exist_ok=True)
ckpt = tf.keras.callbacks.ModelCheckpoint(
    "models/bird_detector_cnn.keras",
    save_best_only=True,
    monitor="val_accuracy"
)

steps_per_epoch   = len(train_files) // BATCH
validation_steps  = len(val_files)   // BATCH

model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    callbacks=[ckpt]
)

print("✅ Trained CNN (best in models/bird_detector_cnn.keras)")
