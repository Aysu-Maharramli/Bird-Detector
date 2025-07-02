#!/usr/bin/env python3
# scripts/extract_features.py

import os
import glob
import yaml
import numpy as np
import pandas as pd
import boto3
import io
import librosa
from tqdm import tqdm


# 0) AWS S3 setup
BUCKET = "bird-detector-aysu-2025" 
s3 = boto3.client("s3")

# 1) Load config
cfg = yaml.safe_load(open("src/config.yaml"))
SR = cfg["audio"]["sample_rate"]
N_MFCC = 13

# 2) Prepare output dirs (you can still save locally if you like)
feat_dir = "data/processed/features"
meta_dir = "data/processed/metadata"
os.makedirs(feat_dir, exist_ok=True)
os.makedirs(meta_dir, exist_ok=True)

rows = []

# 3) List all interim WAV keys in S3
paginator = s3.get_paginator("list_objects_v2")
pages = paginator.paginate(Bucket=BUCKET, Prefix="interim/")
wav_keys = [
    obj["Key"]
    for page in pages
    for obj in page.get("Contents", [])
    if obj["Key"].endswith(".wav")
]

print(f"Found {len(wav_keys)} WAV chunks in s3://{BUCKET}/interim/")

# 4) Process each WAV

for key in tqdm(wav_keys, desc="Extracting features"):
    clip = os.path.basename(key)                            # e.g. “American_Robin_1005778_chunk00.wav”
    species = clip.rsplit("_", 2)[0].replace("_", " ")      # -> “American Robin”
    #print(f"Processing {clip} for species {species}")

    obj = s3.get_object(Bucket=BUCKET, Key=key)
    wav_bytes = obj["Body"].read()
    bio = io.BytesIO(wav_bytes)

    y, _ = librosa.load(bio, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    out_npy = os.path.join(feat_dir, clip.replace(".wav", ".npy"))
    np.save(out_npy, feat)

    rows.append({
        "clip": clip,
        "species": species,
        "label": cfg["species"].index(species),  # now will succeed
        "feature_path": out_npy
    })


# 5) Write metadata CSV
meta_df = pd.DataFrame(rows)
meta_df.to_csv(os.path.join(meta_dir, "metadata.csv"), index=False)
print(f"Extracted features for {len(rows)} clips")
