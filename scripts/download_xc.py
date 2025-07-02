#!/usr/bin/env python3
import os
import csv
import yaml
import requests
import boto3
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor

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

# helper to check if we've already uploaded this species
def already_done(sp):
    species_safe = sp.replace(" ", "_")
    resp = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=f"raw/{species_safe}/",
        MaxKeys=1
    )
    return "Contents" in resp

# this is the per‐species work
def download_one(sp):
    if already_done(sp):
        print(f"✅ SKIP {sp}  (already on S3)")
        return

    print(f"Querying X-C for {sp} …")
    query = quote_plus(sp)
    resp = requests.get(BASE_URL + query).json()
    recs = resp.get("recordings", [])[:MAX_PER_SPECIES]

    raw_meta = []
    species_safe = sp.replace(" ", "_")
    for rec in recs:
        rec_id = rec["id"]
        file_url = rec["file"]
        if file_url.startswith("//"):
            file_url = "https:" + file_url

        # Download bytes & push straight to S3
        data = requests.get(file_url).content
        key = f"raw/{species_safe}/{rec_id}.mp3"
        s3.put_object(Bucket=BUCKET, Key=key, Body=data)
        print(f"  ☁️ uploaded s3://{BUCKET}/{key}")

        raw_meta.append({
            "filename": f"{species_safe}_{rec_id}.mp3",
            "species": sp,
            "call_type": rec.get("type", "unknown")
        })

    # write local metadata CSV for this species
    csv_path = f"data/raw/{species_safe}_meta.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename","species","call_type"])
        writer.writeheader()
        writer.writerows(raw_meta)

    print(f"✅ Finished {sp} ({len(recs)} recordings)")

# 4) Kick off a thread pool to do 8 at once
if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=8) as pool:
        pool.map(download_one, species_list)

    print("🎉 All species processed!")
