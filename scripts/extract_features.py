# scripts/extract_features.py
#!/usr/bin/env python3

import os
import yaml
import numpy as np
import pandas as pd
import boto3
import io
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 0) AWS S3 setup
BUCKET = "bird-detector-aysu-2025"
s3 = boto3.client("s3")

# 1) Load config
tmp_cfg = yaml.safe_load(open("src/config.yaml"))
SR = tmp_cfg["audio"]["sample_rate"]
N_MFCC = 13

# 2) Prepare local dirs
feat_dir = "data/processed/features"
meta_dir = "data/processed/metadata"
os.makedirs(feat_dir, exist_ok=True)
os.makedirs(meta_dir, exist_ok=True)

# 3) List WAV keys in S3
paginator = s3.get_paginator("list_objects_v2")
pages = paginator.paginate(Bucket=BUCKET, Prefix="interim/")
wav_keys = [obj["Key"] for page in pages for obj in page.get("Contents", []) if obj["Key"].endswith(".wav")]
print(f"Found {len(wav_keys)} WAV chunks in s3://{BUCKET}/interim/")

# 4) Define feature extraction
 def extract_one(key):
    clip = os.path.basename(key)
    species = clip.rsplit("_", 2)[0].replace("_", " ")
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    bio = io.BytesIO(obj["Body"].read())
    y, _ = librosa.load(bio, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    out_npy = os.path.join(feat_dir, clip.replace(".wav", ".npy"))
    np.save(out_npy, feat)
    return {"clip": clip, "species": species, "label": tmp_cfg["species"].index(species), "feature_path": out_npy}

# 5) Parallel extraction
if __name__ == '__main__':
    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(extract_one, key) for key in wav_keys]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Extracting features"):
            rows.append(f.result())
    pd.DataFrame(rows).to_csv(os.path.join(meta_dir, "metadata.csv"), index=False)
    print(f"Extracted features for {len(rows)} clips")
