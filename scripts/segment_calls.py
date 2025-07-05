#!/usr/bin/env python3
# scripts/segment_calls.py

import os
import subprocess
import tempfile
import boto3
import soundfile as sf
import yaml
from concurrent.futures import ThreadPoolExecutor

# 0) AWS S3 setup
BUCKET = "bird-detector-aysu-2025"
s3 = boto3.client("s3")

# 1) Load config
cfg = yaml.safe_load(open("src/config.yaml"))
SR = cfg["audio"]["sample_rate"]
CHUNK_DUR = cfg["audio"]["chunk_duration"]  

# 2) List all raw MP3 keys in S3
mp3_keys = []
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=BUCKET, Prefix="raw/"):
    for obj in page.get("Contents", []):
        if obj["Key"].lower().endswith(".mp3"):
            mp3_keys.append(obj["Key"])

print(f"Found {len(mp3_keys)} MP3s in s3://{BUCKET}/raw/")

# 3) Define per-file segmentation
def segment_one(key):
    """
    Download one MP3, convert to WAV, split into chunks,
    and upload them under interim/{species}/.
    """
    parts = key.split("/")
    species_safe = parts[1]
    rec_id_mp3 = parts[2]
    rec_id = os.path.splitext(rec_id_mp3)[0]
    base = f"{species_safe}_{rec_id}"

    # Download MP3
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
        try:
            s3.download_fileobj(BUCKET, key, tmp_mp3)
        except Exception as e:
            print(f"⚠️  failed download {key}: {e}")
            return
        mp3_path = tmp_mp3.name

    # Convert to WAV
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ar", str(SR), wav_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        print(f"⚠️  skipping corrupt file {key}")
        os.remove(mp3_path)
        os.remove(wav_path)
        return

    # Read & chunk
    try:
        y, sr_native = sf.read(wav_path)
    except Exception as e:
        print(f"⚠️  failed to read WAV {wav_path}: {e}")
        os.remove(mp3_path)
        os.remove(wav_path)
        return

    total_secs = len(y) / sr_native
    n_chunks = int(total_secs // CHUNK_DUR)

    for i in range(n_chunks):
        start = int(i * CHUNK_DUR * sr_native)
        end   = int((i + 1) * CHUNK_DUR * sr_native)
        clip = y[start:end]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_chunk:
            sf.write(tmp_chunk.name, clip, sr_native)
            chunk_path = tmp_chunk.name

        chunk_name = f"{base}_chunk{i:02d}.wav"
        s3_key = f"interim/{species_safe}/{chunk_name}"
        with open(chunk_path, "rb") as f:
            s3.put_object(Bucket=BUCKET, Key=s3_key, Body=f)
        print(f"    ⬆️  {s3_key}")

        os.remove(chunk_path)

    # Clean up
    os.remove(mp3_path)
    os.remove(wav_path)


# 4) Parallel execution
if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=8) as pool:
        pool.map(segment_one, mp3_keys)
    print("✅ All done segmenting and uploading!")
