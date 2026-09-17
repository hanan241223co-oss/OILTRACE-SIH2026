"""
OILTRACE Inference Script
Usage:
    python inference.py --input <sar.tif> --model <model.joblib> --output <mask_pred.tif>
    python inference.py --input <sar.tif>   (uses defaults)
"""
import os
import sys
import argparse
import struct
import numpy as np
import tifffile
import joblib
from scipy.ndimage import uniform_filter

# ── feature engineering (inline so this file is self-contained) ──────────────
def _lee_filter(img, size=5):
    m  = uniform_filter(img, (size, size))
    m2 = uniform_filter(img**2, (size, size))
    v  = np.maximum(m2 - m**2, 0.0)
    ov = np.var(img)
    if ov == 0:
        return img
    w = v / (v + ov)
    return m + w * (img - m)

def compute_features(sar_array, apply_lee=True):
    """Return (H, W, 4) float32 feature stack from (H, W, 2) or (2, H, W) SAR input."""
    if sar_array.ndim == 3 and sar_array.shape[0] == 2:
        sar_array = np.transpose(sar_array, (1, 2, 0))
    vv_db = np.clip(sar_array[:, :, 0].astype(np.float32), -50.0, 5.0)
    vh_db = np.clip(sar_array[:, :, 1].astype(np.float32), -50.0, 5.0)
    if apply_lee:
        vv_db = _lee_filter(vv_db)
        vh_db = _lee_filter(vh_db)
    vv_n = np.clip((vv_db - (-45.0)) / 45.0, 0.0, 1.0)
    vh_n = np.clip((vh_db - (-45.0)) / 45.0, 0.0, 1.0)
    diff_n = np.clip(((vv_db - vh_db) - (-10.0)) / 25.0, 0.0, 1.0)
    m  = uniform_filter(vv_n, (5, 5))
    m2 = uniform_filter(vv_n**2, (5, 5))
    std_n = np.clip(np.sqrt(np.maximum(m2 - m**2, 0.0)) / 0.15, 0.0, 1.0)
    return np.stack([vv_n, vh_n, diff_n, std_n], axis=-1).astype(np.float32)

# ── GeoTIFF tag copying (minimal, using tifffile) ────────────────────────────
def _copy_geotags(src_path):
    """Return dict of GeoTIFF-related tag codes→values from source TIFF."""
    geo_tag_codes = {33550, 33922, 34264, 34735, 34736, 34737}
    tags_out = {}
    with tifffile.TiffFile(src_path) as tif:
        for tag in tif.pages[0].tags.values():
            if tag.code in geo_tag_codes:
                tags_out[tag.code] = tag.value
    return tags_out

def run_inference(input_path, model_path, output_path, apply_lee=True, threshold=0.3):
    print(f"Loading SAR image: {input_path}")
    with tifffile.TiffFile(input_path) as tif:
        sar = tif.asarray()

    print(f"SAR shape: {sar.shape}, dtype: {sar.dtype}")
    feats = compute_features(sar, apply_lee=apply_lee)  # (H, W, 4)
    H, W, C = feats.shape

    print(f"Loading model: {model_path}")
    clf = joblib.load(model_path)

    print(f"Running prediction on {H*W:,} pixels …")
    X_flat = feats.reshape(-1, C)
    y_proba = clf.predict_proba(X_flat)[:, 1]   # class-1 (oil) probability
    y_pred  = (y_proba >= threshold).astype(np.uint8)

    pred_map  = y_pred.reshape(H, W)
    proba_map = y_proba.reshape(H, W).astype(np.float32)

    oil_pct = pred_map.mean() * 100
    print(f"Predicted oil coverage: {oil_pct:.2f}%")

    # ── save binary mask ──
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tifffile.imwrite(output_path, pred_map, dtype=np.uint8)
    print(f"Saved binary mask -> {output_path}")

    # -- save probability map alongside --
    proba_path = output_path.replace(".tif", "_proba.tif")
    tifffile.imwrite(proba_path, proba_map)
    print(f"Saved probability map -> {proba_path}")

    return pred_map, proba_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OILTRACE RF Inference")
    parser.add_argument("--input",  required=True,
                        help="Input SAR GeoTIFF (2-channel VV/VH Float32)")
    parser.add_argument("--model",  default=r"D:\OILTRACE\models\best_model.joblib")
    parser.add_argument("--output", default=r"D:\OILTRACE\data\results\detection\pred_mask.tif")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Oil probability threshold (default 0.3)")
    parser.add_argument("--no_lee", action="store_true",
                        help="Skip Lee speckle filter")
    args = parser.parse_args()

    run_inference(args.input, args.model, args.output,
                  apply_lee=not args.no_lee, threshold=args.threshold)
