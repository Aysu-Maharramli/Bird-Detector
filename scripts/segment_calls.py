#!/usr/bin/env python3
# scripts/segment_calls.py

import os
import subprocess
import tempfile

import boto3
import soundfile as sf
import yaml

# 0) AWS S3 setup
BUCKET = "bird-detector-aysu-2025"
s3 = boto3.client("s3")

# 1) Load config
cfg = yaml.safe_load(open("src/config.yaml"))
SR = cfg["audio"]["sample_rate"]
CHUNK_DUR = cfg["audio"]["chunk_duration"]  # in seconds

# 2) List all raw MP3 keys in S3
mp3_keys = []
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=BUCKET, Prefix="raw/"):
    for obj in page.get("Contents", []):
        if obj["Key"].lower().endswith(".mp3"):
            mp3_keys.append(obj["Key"])

print(f"Found {len(mp3_keys)} MP3s in s3://{BUCKET}/raw/")

# 3) Process each MP3
for key in mp3_keys:
    # key format: raw/Species_Name/12345.mp3
    parts = key.split("/")
    species_safe = parts[1]             # e.g. "American_Robin"
    rec_id_mp3 = parts[2]               # e.g. "12345.mp3"
    rec_id = os.path.splitext(rec_id_mp3)[0]
    base = f"{species_safe}_{rec_id}"   # e.g. "American_Robin_12345"
    print(f"🔊 Segmenting {key}")

    # 3a) Download MP3 to a temp file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
        s3.download_fileobj(BUCKET, key, tmp_mp3)
        tmp_mp3_path = tmp_mp3.name

    # 3b) Convert MP3 -> WAV at SR via ffmpeg into another temp file
    tmp_wav_fd, tmp_wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_wav_fd)
    subprocess.run([
        "ffmpeg", "-y",
        "-i", tmp_mp3_path,
        "-ar", str(SR),
        tmp_wav_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 3c) Read WAV and split into chunks
    y, sr_native = sf.read(tmp_wav_path)
    total_secs = len(y) / sr_native
    n_chunks = int(total_secs // CHUNK_DUR)

    for i in range(n_chunks):
        start = int(i * CHUNK_DUR * sr_native)
        end   = int((i + 1) * CHUNK_DUR * sr_native)
        clip = y[start:end]

        # write chunk to a temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_chunk:
            sf.write(tmp_chunk.name, clip, sr_native)
            tmp_chunk_path = tmp_chunk.name

        # upload chunk to S3 under interim/
        chunk_name = f"{base}_chunk{i:02d}.wav"
        s3_key = f"interim/{species_safe}/{chunk_name}"
        with open(tmp_chunk_path, "rb") as data:
            s3.put_object(Bucket=BUCKET, Key=s3_key, Body=data)
        print(f"    ⬆️  uploaded chunk {i+1}/{n_chunks}: s3://{BUCKET}/{s3_key}")

        os.remove(tmp_chunk_path)

    # 3d) Clean up temp MP3 and WAV
    os.remove(tmp_mp3_path)
    os.remove(tmp_wav_path)

print("✅ All done segmenting and uploading!")
