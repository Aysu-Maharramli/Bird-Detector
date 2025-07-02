# scripts/segment_calls.py
#!/usr/bin/env python3

import os
import subprocess
import tempfile
import boto3
import soundfile as sf
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

# 0) AWS S3 setup
BUCKET = "bird-detector-aysu-2025"
s3 = boto3.client("s3")

# 1) Load config
tmp_cfg = yaml.safe_load(open("src/config.yaml"))
SR = tmp_cfg["audio"]["sample_rate"]
CHUNK_DUR = tmp_cfg["audio"]["chunk_duration"]  # seconds per clip

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
    parts = key.split("/")
    species_safe = parts[1]
    rec_id = os.path.splitext(parts[2])[0]
    base = f"{species_safe}_{rec_id}"

    # Download MP3 → temp file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
        s3.download_fileobj(BUCKET, key, tmp_mp3)
        tmp_mp3_path = tmp_mp3.name

    # Convert MP3 → WAV at SR
    tmp_wav_fd, tmp_wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_wav_fd)
    subprocess.run([
        "ffmpeg", "-y",
        "-i", tmp_mp3_path,
        "-ar", str(SR), tmp_wav_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Read & split
    y, sr_native = sf.read(tmp_wav_path)
    n_chunks = int(len(y) / sr_native // CHUNK_DUR)
    for i in range(n_chunks):
        start = int(i * CHUNK_DUR * sr_native)
        end = start + int(CHUNK_DUR * sr_native)
        clip = y[start:end]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_chunk:
            sf.write(tmp_chunk.name, clip, sr_native)
            tmp_chunk_path = tmp_chunk.name

        chunk_name = f"{base}_chunk{i:02d}.wav"
        s3_key = f"interim/{species_safe}/{chunk_name}"
        with open(tmp_chunk_path, "rb") as data:
            s3.put_object(Bucket=BUCKET, Key=s3_key, Body=data)
        os.remove(tmp_chunk_path)

    # Cleanup
    os.remove(tmp_mp3_path)
    os.remove(tmp_wav_path)
    return key

# 4) Parallel execution
if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed(pool.map(lambda k: segment_one(k), mp3_keys)):
            print(f"Segmented: {future}")
    print("✅ All done segmenting and uploading!")