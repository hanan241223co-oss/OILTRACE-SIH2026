"""
OILTRACE Dataset Preparation
Dataset: Sentinel-1 SAR Oil Spill Image Dataset (Parts I & II)
Source: zenodo.org/records/8346860 (Part I - Oil Spill images + masks)
        zenodo.org/records/8253899 (Part II - No Oil + Lookalike images + masks)
License: Creative Commons Attribution 4.0 International (CC-BY-4.0)
Authors: Trujillo-Acatitla, Rubicel; Tuxpan-Vargas, Jose (et al.)

Goal: Download 50 oil-spill, 50 no-oil, 50 look-alike images + masks
      and organize them under D:\OILTRACE\data\
"""

import os
import sys
import subprocess
import shutil
import csv
import io
import urllib.request
import struct
import hashlib
import zlib
import py7zr
import numpy as np
import tifffile

# === PATHS ===
DATA_DIR = r"D:\OILTRACE\data"
SCRATCH = r"C:\Users\AL-Taif\.gemini\antigravity\brain\a530a480-8af3-4ffb-98de-936953f19fc2\scratch"

# === OUTPUT STRUCTURE ===
CLASSES = {
    "oil_spill": {
        "img_dir": os.path.join(DATA_DIR, "images", "oil_spill"),
        "mask_dir": os.path.join(DATA_DIR, "masks", "oil_spill"),
        "img_url": "https://zenodo.org/records/8346860/files/01_Train_Val_Oil_Spill_images.7z",
        "img_size": 40712942245,
        "img_prefix": "Oil/",
        "mask_local_dir": os.path.join(SCRATCH, "Mask_oil"),
        "mask_url": "https://zenodo.org/records/8346860/files/01_Train_Val_Oil_Spill_mask.7z",
        "count": 50,
    },
    "no_oil": {
        "img_dir": os.path.join(DATA_DIR, "images", "no_oil"),
        "mask_dir": os.path.join(DATA_DIR, "masks", "no_oil"),
        "img_url": "https://zenodo.org/records/8253899/files/01_Train_Val_No_Oil_Images.7z",
        "img_size": 22931223979,
        "img_prefix": "No_Oil/",
        "mask_local_dir": os.path.join(SCRATCH, "Mask_no_oil"),
        "count": 50,
    },
    "look_alike": {
        "img_dir": os.path.join(DATA_DIR, "images", "look_alike"),
        "mask_dir": os.path.join(DATA_DIR, "masks", "look_alike"),
        "img_url": "https://zenodo.org/records/8253899/files/01_Train_Val_Lookalike_images.7z",
        "img_size": 22993852696,
        "img_prefix": "Lookalike/",
        "mask_local_dir": os.path.join(SCRATCH, "Mask_lookalike"),
        "count": 50,
    },
}

# === CREATE OUTPUT DIRS ===
for cls_info in CLASSES.values():
    os.makedirs(cls_info["img_dir"], exist_ok=True)
    os.makedirs(cls_info["mask_dir"], exist_ok=True)
print("Output directories created.")

# === HTTP RANGE READER ===
class HttpRangeReader(io.RawIOBase):
    def __init__(self, url, size, block_size=20 * 1024 * 1024):
        self.url = url
        self.size = size
        self.pos = 0
        self.bytes_read = 0
        self.requests_count = 0
        self.block_size = block_size
        self.buffer = b""
        self.buffer_start = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        if size == -1 or size is None:
            size = self.size - self.pos
        if self.pos >= self.size or size <= 0:
            return b""

        # Check buffer
        if (self.buffer and self.buffer_start <= self.pos < 
                self.buffer_start + len(self.buffer)):
            offset_in = self.pos - self.buffer_start
            avail = len(self.buffer) - offset_in
            if avail >= size:
                chunk = self.buffer[offset_in:offset_in + size]
                self.pos += len(chunk)
                return chunk

        # Fetch new block
        fetch_size = max(size, self.block_size)
        end = min(self.pos + fetch_size - 1, self.size - 1)
        req = urllib.request.Request(self.url, headers={
            "User-Agent": "curl/7.68.0",
            "Range": f"bytes={self.pos}-{end}"
        })
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
                    self.buffer = data
                    self.buffer_start = self.pos
                    chunk = data[:size]
                    self.pos += len(chunk)
                    self.bytes_read += len(data)
                    self.requests_count += 1
                    return chunk
            except Exception as e:
                if attempt == 2:
                    raise
        return b""


def get_archive_filenames(url, size):
    """Get all filenames from a 7z archive using HTTP range reads."""
    reader = HttpRangeReader(url, size)
    with py7zr.SevenZipFile(reader, mode='r') as z:
        names = z.getnames()
    return sorted(names)


def extract_files_from_archive(url, size, targets, out_dir):
    """Extract specific files from a remote 7z using HTTP range reads."""
    reader = HttpRangeReader(url, size)
    with py7zr.SevenZipFile(reader, mode='r') as z:
        z.extract(path=out_dir, targets=targets)
    return reader.bytes_read, reader.requests_count


# === STEP 1: Get file lists and select 50 from each class ===
print("\n=== Getting archive file lists ===")

results = {}
for cls_name, cls_info in CLASSES.items():
    print(f"\n[{cls_name}] Listing {cls_info['img_url']}...")
    try:
        names = get_archive_filenames(cls_info["img_url"], cls_info["img_size"])
        tif_names = [n for n in names if n.endswith(".tif") and "/" in n]
        print(f"  Found {len(tif_names)} TIF files in archive")
        # Select first 50 in sorted order
        selected = sorted(tif_names)[:cls_info["count"]]
        results[cls_name] = selected
        print(f"  Selected {len(selected)} for download")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n=== Downloading selected image TIFs ===")

total_bytes_downloaded = 0
download_stats = {}

for cls_name, selected_files in results.items():
    cls_info = CLASSES[cls_name]
    img_out = cls_info["img_dir"]
    
    print(f"\n[{cls_name}] Downloading {len(selected_files)} files to {img_out}...")
    
    bytes_dl, n_reqs = extract_files_from_archive(
        cls_info["img_url"], cls_info["img_size"], selected_files, img_out
    )
    total_bytes_downloaded += bytes_dl
    download_stats[cls_name] = {"bytes": bytes_dl, "requests": n_reqs, "count": len(selected_files)}
    
    # Move files from subdirectory to cls img_dir
    prefix = cls_info["img_prefix"].rstrip("/")
    src_subdir = os.path.join(img_out, prefix)
    if os.path.isdir(src_subdir):
        for f in os.listdir(src_subdir):
            src = os.path.join(src_subdir, f)
            dst = os.path.join(img_out, f)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        os.rmdir(src_subdir)
    
    actual_count = len([f for f in os.listdir(img_out) if f.endswith(".tif")])
    print(f"  Downloaded {actual_count} TIFs, {bytes_dl / 1024 / 1024:.2f} MB transferred in {n_reqs} requests")

print(f"\nTotal downloaded: {total_bytes_downloaded / 1024 / 1024:.2f} MB")
print("Download complete. Saving stats...")

# Save stats to file
stats_path = os.path.join(DATA_DIR, "download_stats.txt")
with open(stats_path, "w") as f:
    for cls, stats in download_stats.items():
        f.write(f"{cls}: {stats['count']} files, {stats['bytes']/1024/1024:.2f} MB, {stats['requests']} requests\n")
    f.write(f"Total: {total_bytes_downloaded/1024/1024:.2f} MB\n")
print(f"Stats saved to {stats_path}")
