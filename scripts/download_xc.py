#!/usr/bin/env python3
# scripts/download_xc.py

import os
import csv
import yaml
import requests
import boto3
from urllib.parse import quote_plus

# 0) AWS S3 setup
BUCKET = "bird-detector-aysu-2025"    
s3 = boto3.client("s3")

# 1) Load our species list
cfg = yaml.safe_load(open("src/config.yaml"))
species_list = cfg["species"]

# 2) Set constants
BASE_URL = "https://www.xeno-canto.org/api/2/recordings?query="
MAX_PER_SPECIES = 100

# 3) Prepare local metadata CSV folder
os.makedirs("data/raw", exist_ok=True)
raw_meta = []

# 4) Loop over species and download
for sp in species_list:
    query = quote_plus(sp)
    url = BASE_URL + query
    print(f"Querying Xeno-Canto for {sp} → {url}")
    resp = requests.get(url).json()
    recordings = resp.get("recordings", [])[:MAX_PER_SPECIES]

    for rec in recordings:
        rec_id = rec["id"]
        species_safe = sp.replace(" ", "_")
        filename = f"{species_safe}_{rec_id}.mp3"
        s3_key = f"raw/{species_safe}/{rec_id}.mp3"

        # 4a) Download the MP3 bytes
        file_url = rec["file"]
        if file_url.startswith("//"):
            file_url = "https:" + file_url
        print(f"  → downloading {file_url}")
        data = requests.get(file_url).content

        # 4b) Upload directly to S3 (no local disk write)
        s3.put_object(Bucket=BUCKET, Key=s3_key, Body=data)
        print(f"    ☁️  uploaded to s3://{BUCKET}/{s3_key}")

        # 4c) Record metadata
        raw_meta.append({
            "filename": filename,
            "species": sp,
            "call_type": rec.get("type", "unknown")
        })

    print(f"✅ Finished species: {sp} ({len(recordings)} recordings)")

# 5) Write out the raw metadata CSV locally
csv_path = "data/raw/raw_metadata.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "species", "call_type"])
    writer.writeheader()
    writer.writerows(raw_meta)

print(f"📄 Wrote raw metadata: {csv_path}")
print("🎉 All done!")
