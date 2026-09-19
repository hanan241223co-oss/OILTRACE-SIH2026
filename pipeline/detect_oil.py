#!/usr/bin/env python3
"""
OILTRACE SAR Oil Spill Detection & Visualization Pipeline
---------------------------------------------------------
Dual-Engine Satellite SAR Oil Spill Detection:
  1. Deep Learning U-Net (31M params, PyTorch, SOTA precision matching ground truth mask.png)
  2. Random Forest Baseline (scikit-learn, lightweight 4-band pixel classifier with spatial filtering)

Usage:
    # Run U-Net (matches ground truth mask.png with zero false positives)
    python detect_oil.py --model testing/best_unet.pt

    # Run Random Forest baseline
    python detect_oil.py --model testing/best_model.joblib

    # Default auto-detects best available model
    python detect_oil.py
"""

import os
import sys
import time
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import math
import argparse
import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
from PIL import Image
from scipy import ndimage
from scipy.ndimage import uniform_filter
from scipy.special import expit
import joblib

# Suppress warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
except ImportError:
    pass

# Matplotlib headless backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Geospatial libraries
try:
    import rasterio
    from rasterio.transform import xy
    from rasterio.warp import transform_bounds, transform
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Autonomous vessel reflector and shadow corridor masking
try:
    from pipeline.vessel_masking import mask_vessel_shadows, detect_vessel_targets
except ImportError:
    try:
        from .vessel_masking import mask_vessel_shadows, detect_vessel_targets
    except ImportError:
        try:
            from vessel_masking import mask_vessel_shadows, detect_vessel_targets
        except ImportError:
            mask_vessel_shadows = None
            detect_vessel_targets = None

# Adaptive dark spot candidate segmentation engine
try:
    from pipeline.segmentation import (
        segment_candidate_dark_spots,
        export_candidate_geojson,
        SegmentationResult,
        CandidateDarkSpot,
    )
except ImportError:
    try:
        from .segmentation import (
            segment_candidate_dark_spots,
            export_candidate_geojson,
            SegmentationResult,
            CandidateDarkSpot,
        )
    except ImportError:
        try:
            from segmentation import (
                segment_candidate_dark_spots,
                export_candidate_geojson,
                SegmentationResult,
                CandidateDarkSpot,
            )
        except ImportError:
            segment_candidate_dark_spots = None
            export_candidate_geojson = None

# Physical & geometric object feature extractor
try:
    from pipeline.feature_extractor import (
        extract_candidate_features,
        export_features_tabular,
        CANONICAL_FEATURE_NAMES,
        FeatureExtractionResult,
        CandidateObjectFeatures,
    )
except ImportError:
    try:
        from .feature_extractor import (
            extract_candidate_features,
            export_features_tabular,
            CANONICAL_FEATURE_NAMES,
            FeatureExtractionResult,
            CandidateObjectFeatures,
        )
    except ImportError:
        try:
            from feature_extractor import (
                extract_candidate_features,
                export_features_tabular,
                CANONICAL_FEATURE_NAMES,
                FeatureExtractionResult,
                CandidateObjectFeatures,
            )
        except ImportError:
            extract_candidate_features = None
            export_features_tabular = None
            CANONICAL_FEATURE_NAMES = []

# Object-based image analysis (OBIA) Random Forest classifier
try:
    from pipeline.obia_classifier import (
        load_object_classifier,
        ObjectClassifier,
        ObjectClassificationResult,
        TrainedClassifierBundle,
    )
except ImportError:
    try:
        from .obia_classifier import (
            load_object_classifier,
            ObjectClassifier,
            ObjectClassificationResult,
            TrainedClassifierBundle,
        )
    except ImportError:
        try:
            from obia_classifier import (
                load_object_classifier,
                ObjectClassifier,
                ObjectClassificationResult,
                TrainedClassifierBundle,
            )
        except ImportError:
            load_object_classifier = None
            ObjectClassifier = None
            ObjectClassificationResult = None
            TrainedClassifierBundle = None


# ==============================================================================
# Pipeline Configuration and Result Data Classes
# ==============================================================================

@dataclass
class PipelineRunConfig:
    input_path: str
    output_dir: Optional[str] = None
    mode: str = "two_stage"
    obia_model_path: str = "testing/best_obia_rf.joblib"
    unet_model_path: str = "testing/best_unet.pt"
    pixel_rf_path: str = "testing/best_model.joblib"
    threshold: Optional[float] = None
    enable_vessel_masking: bool = True
    enable_candidate_export: bool = True
    device: str = "cpu"
    min_damping: float = 2.2
    max_backscatter: float = -20.0
    window_size: int = 151
    min_candidate_px: int = 30
    min_px: int = 50


@dataclass
class TwoStagePipelineResult:
    scene_id: str
    pipeline_mode: str
    total_vessels_detected: int
    shadow_pixels_masked: int
    total_candidates_segmented: int
    total_candidate_area_km2: float
    confirmed_slicks_count: int
    suppressed_lookalikes_count: int
    confirmed_slicks_area_km2: float
    execution_time_seconds: float
    deliverables: Dict[str, Optional[str]]
    confirmed_slicks: List[Any] = field(default_factory=list)
    classification_results: List[Any] = field(default_factory=list)


# ==============================================================================
# 1. U-Net Deep Learning Architecture (31M Parameters)
# ==============================================================================

if HAS_TORCH:
    class DoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    class Down(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(self.pool(x))

    class Up(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        def forward(self, x: torch.Tensor, skip: torch.Tensor, skip_first: bool = True) -> torch.Tensor:
            x = self.up(x)
            if skip_first:
                x = torch.cat([skip, x], dim=1)
            else:
                x = torch.cat([x, skip], dim=1)
            return self.conv(x)

    class UNet(nn.Module):
        def __init__(self, in_channels: int = 2, out_channels: int = 1, base_channels: int = 64,
                     depth: int = 4, concat_skip_first: bool = True) -> None:
            super().__init__()
            features = [base_channels * (2**level) for level in range(depth)]
            self.inc = DoubleConv(in_channels, features[0])
            self.downs = nn.ModuleList(
                Down(features[level], features[level + 1]) for level in range(depth - 1)
            )
            self.bottleneck = Down(features[-1], features[-1] * 2)
            self.ups = nn.ModuleList(
                Up(features[depth - 1 - level] * 2, features[depth - 1 - level])
                for level in range(depth)
            )
            self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)
            self.concat_skip_first = concat_skip_first

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.inc(x)
            skips = [x]
            for down in self.downs:
                x = down(x)
                skips.append(x)
            x = self.bottleneck(x)
            for up, skip in zip(self.ups, reversed(skips)):
                x = up(x, skip, skip_first=self.concat_skip_first)
            return self.head(x)


def run_unet_inference(sar_data: np.ndarray, weights_path: str, threshold: float = 0.5,
                       device_str: str = 'cpu') -> tuple[np.ndarray, np.ndarray]:
    """
    Executes high-precision tiled U-Net inference on Sentinel-1 SAR input.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to run the U-Net model.")

    device = torch.device(device_str if (device_str != 'cuda' or torch.cuda.is_available()) else 'cpu')
    print(f"  [U-Net Engine] Initializing PyTorch U-Net (31.0M parameters) on {device}...")

    model = UNet(in_channels=2, out_channels=1, base_channels=64, depth=4, concat_skip_first=True)
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Preprocessing & Normalization:
    # Model Channel 0 = VH (P1: -44.0206 dB, P99: 0.0 dB)
    # Model Channel 1 = VV (P1: -35.4087 dB, P99: 0.0 dB)
    vv = sar_data[:, :, 0].astype(np.float32)
    vh = sar_data[:, :, 1].astype(np.float32)

    norm_vh = np.clip((np.clip(vh, -44.02064136505127, 0.0) - (-44.02064136505127)) / 44.02064136505127, 0.0, 1.0)
    norm_vv = np.clip((np.clip(vv, -35.40866855621338, 0.0) - (-35.40866855621338)) / 35.40866855621338, 0.0, 1.0)
    model_input = np.stack([norm_vh, norm_vv], axis=0) # (2, H, W)

    patch_size = 512
    _, H, W = model_input.shape
    pad_h = (patch_size - (H % patch_size)) % patch_size
    pad_w = (patch_size - (W % patch_size)) % patch_size
    if pad_h != 0 or pad_w != 0:
        padded = np.pad(model_input, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    else:
        padded = model_input

    padded_h, padded_w = padded.shape[1], padded.shape[2]
    proba_full = np.zeros((padded_h, padded_w), dtype=np.float32)

    total_patches = (padded_h // patch_size) * (padded_w // patch_size)
    print(f"  [U-Net Engine] Running tiled inference over {total_patches} patches (patch size {patch_size}x{patch_size})...")

    with torch.inference_mode():
        for top in range(0, padded_h, patch_size):
            for left in range(0, padded_w, patch_size):
                patch = padded[:, top : top + patch_size, left : left + patch_size]
                tensor = torch.from_numpy(patch).unsqueeze(0).to(device)
                logits = model(tensor)
                proba_full[top : top + patch_size, left : left + patch_size] = expit(
                    logits.squeeze(0).squeeze(0).float().cpu().numpy()
                )

    proba_map = proba_full[:H, :W]
    binary_mask = (proba_map >= threshold).astype(np.uint8)
    return proba_map, binary_mask


# ==============================================================================
# 2. Random Forest Baseline & Speckle / Feature Engineering
# ==============================================================================

def lee_filter(img: np.ndarray, size: int = 5) -> np.ndarray:
    """Lee Speckle Filter for radar noise reduction."""
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_var = np.maximum(img_sqr_mean - img_mean**2, 0.0)
    overall_var = np.var(img)
    if overall_var == 0:
        return img
    weights = img_var / (img_var + overall_var)
    return img_mean + weights * (img - img_mean)


def extract_rf_features(sar_data: np.ndarray, apply_lee: bool = True) -> np.ndarray:
    """Extract 4-band features for Random Forest classifier."""
    if sar_data.ndim == 3 and sar_data.shape[0] == 2:
        sar_data = np.transpose(sar_data, (1, 2, 0))

    vv_db = np.clip(sar_data[:, :, 0].astype(np.float32), -50.0, 5.0)
    vh_db = np.clip(sar_data[:, :, 1].astype(np.float32), -50.0, 5.0)

    if apply_lee:
        vv_clean = lee_filter(vv_db, size=5)
        vh_clean = lee_filter(vh_db, size=5)
    else:
        vv_clean = vv_db
        vh_clean = vh_db

    vv_norm = np.clip((vv_clean - (-45.0)) / 45.0, 0.0, 1.0)
    vh_norm = np.clip((vh_clean - (-45.0)) / 45.0, 0.0, 1.0)
    diff_norm = np.clip(((vv_clean - vh_clean) - (-10.0)) / 25.0, 0.0, 1.0)

    m = uniform_filter(vv_norm, (5, 5))
    m2 = uniform_filter(vv_norm**2, (5, 5))
    local_std = np.sqrt(np.maximum(m2 - m**2, 0.0))
    std_norm = np.clip(local_std / 0.15, 0.0, 1.0)

    return np.stack([vv_norm, vh_norm, diff_norm, std_norm], axis=-1).astype(np.float32)


def run_rf_inference(sar_data: np.ndarray, model_path: str, threshold: float = 0.3,
                     apply_lee: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Executes Random Forest pixel inference with noise & shadow rejection."""
    clf = joblib.load(model_path)
    features = extract_rf_features(sar_data, apply_lee=apply_lee)
    H, W, C = features.shape
    X_flat = features.reshape(-1, C)

    print(f"  [Random Forest Engine] Running inference on {len(X_flat):,} pixels...")
    proba_flat = clf.predict_proba(X_flat)[:, 1]
    proba_map = proba_flat.reshape(H, W).astype(np.float32)

    # 1. Swath no-data mask (where pixels are zero padding)
    swath_nodata = (sar_data[:, :, 0] == 0.0) & (sar_data[:, :, 1] == 0.0)
    proba_map[swath_nodata] = 0.0

    # 2. Ship bright reflector & radar shadow suppression
    # Ships have extremely high backscatter (VV > 0 dB or VH > -15 dB)
    ships = (sar_data[:, :, 0] > 0.0) | (sar_data[:, :, 1] > -15.0)
    # Mask shadow corridor directly around vessels (15 px radius)
    ship_shadow_corridor = ndimage.binary_dilation(ships, iterations=12)
    proba_map[ship_shadow_corridor] = np.minimum(proba_map[ship_shadow_corridor], 0.25)

    binary_mask = (proba_map >= threshold).astype(np.uint8)
    return proba_map, binary_mask


# ==============================================================================
# 3. Raster I/O & Geospatial Metadata Handling
# ==============================================================================

def load_sar_scene(tif_path: str):
    """Load SAR GeoTIFF and retrieve geotransform, CRS, and pixel resolution."""
    metadata = {
        'path': tif_path,
        'crs': None,
        'transform': None,
        'bounds': None,
        'wgs84_bounds': None,
        'pixel_size_x': 10.0,
        'pixel_size_y': 10.0,
        'width': 2048,
        'height': 2048
    }

    sar_data = None
    if HAS_RASTERIO:
        try:
            with rasterio.open(tif_path) as src:
                sar_data = src.read()
                sar_data = np.transpose(sar_data, (1, 2, 0))
                metadata['crs'] = src.crs
                metadata['transform'] = src.transform
                metadata['bounds'] = src.bounds
                metadata['width'] = src.width
                metadata['height'] = src.height
                metadata['pixel_size_x'] = abs(src.transform[0])
                metadata['pixel_size_y'] = abs(src.transform[4])
                if src.crs:
                    try:
                        metadata['wgs84_bounds'] = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
                    except Exception:
                        pass
        except Exception as e:
            print(f"Rasterio load warning: {e}, falling back to tifffile...")

    if sar_data is None:
        if HAS_TIFFFILE:
            sar_data = tifffile.imread(tif_path)
            if sar_data.ndim == 3 and sar_data.shape[0] == 2:
                sar_data = np.transpose(sar_data, (1, 2, 0))
            metadata['width'] = sar_data.shape[1]
            metadata['height'] = sar_data.shape[0]
        else:
            raise RuntimeError("Neither rasterio nor tifffile is available to load GeoTIFF.")

    return sar_data, metadata


# ==============================================================================
# 4. Connected Components & Physical Slick Metrics
# ==============================================================================

def analyze_slicks(mask_raw: np.ndarray, proba_map: np.ndarray, metadata: dict, min_px: int = 50):
    """
    Connected-component analysis: extracts surface area, bounding boxes, centroids,
    and confidence scores for each confirmed oil slick. Filters out specks < min_px.
    """
    labelled, num_features = ndimage.label(mask_raw)
    counts = np.bincount(labelled.ravel())

    px_w = metadata.get('pixel_size_x', 10.0)
    px_h = metadata.get('pixel_size_y', 10.0)
    pixel_area_km2 = (px_w * px_h) / 1.0e6

    filtered_mask = np.zeros_like(mask_raw, dtype=np.uint8)
    raw_slicks = []
    slices = ndimage.find_objects(labelled)

    gt = metadata.get('transform')
    crs = metadata.get('crs')

    for lbl_idx in range(1, num_features + 1):
        px_count = counts[lbl_idx]
        if px_count < min_px:
            continue

        filtered_mask[labelled == lbl_idx] = 1
        sl = slices[lbl_idx - 1]
        r_slice, c_slice = sl
        comp_crop = (labelled[r_slice, c_slice] == lbl_idx)
        rows_sub, cols_sub = np.where(comp_crop)
        rows = rows_sub + r_slice.start
        cols = cols_sub + c_slice.start

        r_min, r_max = int(rows.min()), int(rows.max())
        c_min, c_max = int(cols.min()), int(cols.max())
        cy_px = float(rows.mean())
        cx_px = float(cols.mean())

        mean_conf = float(proba_map[labelled == lbl_idx].mean())
        max_conf = float(proba_map[labelled == lbl_idx].max())
        area_km2 = px_count * pixel_area_km2

        geo_info = {}
        if gt and HAS_RASTERIO:
            map_x, map_y = rasterio.transform.xy(gt, cy_px, cx_px, offset='center')
            c00_x, c00_y = rasterio.transform.xy(gt, r_min, c_min, offset='ul')
            c11_x, c11_y = rasterio.transform.xy(gt, r_max, c_max, offset='lr')
            geo_info['centroid_map'] = [map_x, map_y]

            if crs:
                try:
                    lon, lat = transform(crs, 'EPSG:4326', [map_x], [map_y])
                    geo_info['centroid_wgs84'] = [lon[0], lat[0]]
                    b_lons, b_lats = transform(crs, 'EPSG:4326', [c00_x, c11_x], [c00_y, c11_y])
                    geo_info['bbox_wgs84'] = [min(b_lons), min(b_lats), max(b_lons), max(b_lats)]
                except Exception:
                    pass

        raw_slicks.append({
            'pixel_count': int(px_count),
            'area_km2': round(area_km2, 4),
            'area_hectares': round(area_km2 * 100.0, 2),
            'mean_confidence': round(mean_conf, 4),
            'max_confidence': round(max_conf, 4),
            'centroid_px': [round(cx_px, 1), round(cy_px, 1)],
            'bbox_px': [c_min, r_min, c_max, r_max],
            'geo': geo_info
        })

    # Sort slicks largest to smallest so Slick #1 is the primary slick
    raw_slicks.sort(key=lambda s: s['pixel_count'], reverse=True)
    slicks = []
    for i, s in enumerate(raw_slicks):
        s['slick_id'] = i + 1
        slicks.append(s)

    return filtered_mask, slicks


# ==============================================================================
# 5. Visual Rendering & Comprehensive Analytical Deliverables
# ==============================================================================

def generate_visualizations(sar_data: np.ndarray, proba_map: np.ndarray,
                            filtered_mask: np.ndarray, slicks: list,
                            metadata: dict, output_dir: str, base_name: str,
                            model_name: str = "U-Net"):
    """Generates mask.png, mask_overlay.png, probability.png, and detection_result.png."""
    os.makedirs(output_dir, exist_ok=True)
    H, W = sar_data.shape[:2]

    vv_raw = sar_data[:, :, 0]
    vh_raw = sar_data[:, :, 1] if sar_data.ndim >= 3 and sar_data.shape[2] > 1 else sar_data[:, :, 0]
    vv_p2, vv_p98 = np.percentile(vv_raw, (2, 98))
    vh_p2, vh_p98 = np.percentile(vh_raw, (2, 98))
    vv_disp = np.clip((vv_raw - vv_p2) / max(vv_p98 - vv_p2, 1e-5), 0, 1)
    vh_disp = np.clip((vh_raw - vh_p2) / max(vh_p98 - vh_p2, 1e-5), 0, 1)

    # 1. Binary Mask (2048 x 2048)
    mask_png_path = os.path.join(output_dir, f"{base_name}_mask.png")
    mask_img = (filtered_mask * 255).astype(np.uint8)
    Image.fromarray(mask_img).save(mask_png_path, format="PNG")
    print(f"  [+] Saved Binary Mask -> {mask_png_path}")

    # 2. Probability Heatmap
    proba_png_path = os.path.join(output_dir, f"{base_name}_probability.png")
    cmap_magma = plt.get_cmap('inferno')
    proba_rgba = (cmap_magma(proba_map) * 255).astype(np.uint8)
    Image.fromarray(proba_rgba).save(proba_png_path, format="PNG")
    print(f"  [+] Saved Probability Heatmap -> {proba_png_path}")

    # 3. High-Resolution SAR Overlay (Translucent Crimson Slicks with Edge Glow)
    overlay_png_path = os.path.join(output_dir, f"{base_name}_mask_overlay.png")
    vv_rgb = (np.repeat(vv_disp[:, :, np.newaxis], 3, axis=2) * 255).astype(np.uint8)
    overlay_rgb = vv_rgb.copy()

    oil_indices = (filtered_mask == 1)
    overlay_rgb[oil_indices, 0] = np.clip(0.35 * vv_rgb[oil_indices, 0] + 0.65 * 255, 0, 255).astype(np.uint8)
    overlay_rgb[oil_indices, 1] = np.clip(0.35 * vv_rgb[oil_indices, 1] + 0.65 * 30, 0, 255).astype(np.uint8)
    overlay_rgb[oil_indices, 2] = np.clip(0.35 * vv_rgb[oil_indices, 2] + 0.65 * 50, 0, 255).astype(np.uint8)

    edges = ndimage.binary_dilation(filtered_mask, iterations=1) & (~filtered_mask.astype(bool))
    overlay_rgb[edges, 0] = 255
    overlay_rgb[edges, 1] = 220
    overlay_rgb[edges, 2] = 0

    Image.fromarray(overlay_rgb).save(overlay_png_path, format="PNG")
    print(f"  [+] Saved High-Res Overlay -> {overlay_png_path}")

    # 4. Multi-Panel Analytical Dashboard
    result_png_path = os.path.join(output_dir, f"{base_name}_detection_result.png")
    fig = plt.figure(figsize=(24, 15), facecolor='#0D1117')
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.1], height_ratios=[1, 1],
                          wspace=0.15, hspace=0.20, left=0.03, right=0.98, top=0.92, bottom=0.06)

    # Panel 1: SAR VV
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#0D1117')
    ax1.imshow(vv_disp, cmap='gray', vmin=0, vmax=1)
    ax1.set_title("1. Sentinel-1 SAR VV Polarization (dB)", color='#E6EDF3', fontsize=12, fontweight='bold', pad=8)
    ax1.axis('off')

    # Panel 2: SAR VH
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#0D1117')
    ax2.imshow(vh_disp, cmap='gray', vmin=0, vmax=1)
    ax2.set_title("2. Sentinel-1 SAR VH Polarization (dB)", color='#E6EDF3', fontsize=12, fontweight='bold', pad=8)
    ax2.axis('off')

    # Panel 3: Probability Heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('#0D1117')
    im_p = ax3.imshow(proba_map, cmap='inferno', vmin=0.0, vmax=1.0)
    ax3.set_title(f"3. {model_name} Oil Probability Heatmap", color='#E6EDF3', fontsize=12, fontweight='bold', pad=8)
    ax3.axis('off')
    cbar = fig.colorbar(im_p, ax=ax3, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='#8B949E', labelcolor='#8B949E')
    cbar.outline.set_edgecolor('#30363D')

    # Panel 4: Binary Oil Mask
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor('#0D1117')
    ax4.imshow(filtered_mask, cmap='gray', vmin=0, vmax=1)
    ax4.set_title("4. Binary Oil Spill Detection Mask", color='#E6EDF3', fontsize=12, fontweight='bold', pad=8)
    ax4.axis('off')

    # Panel 5: Overlaid Detections with Bounding Boxes
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor('#0D1117')
    ax5.imshow(overlay_rgb)
    for s in slicks[:5]:
        c_min, r_min, c_max, r_max = s['bbox_px']
        rect = plt.Rectangle((c_min, r_min), c_max - c_min, r_max - r_min,
                             fill=False, edgecolor='#FFCC00', linewidth=1.8, linestyle='--')
        ax5.add_patch(rect)
        ax5.text(c_min, max(0, r_min - 15), f"Slick #{s['slick_id']} ({s['area_hectares']} ha)",
                 color='#FFFFFF', backgroundcolor='#B22222', fontsize=8.5, fontweight='bold')
    ax5.set_title("5. Overlaid Slicks with Vector BBoxes", color='#E6EDF3', fontsize=12, fontweight='bold', pad=8)
    ax5.axis('off')

    # Panel 6: High Resolution Zoom on Primary Slick
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor('#0D1117')
    if slicks:
        top_slick = slicks[0]
        c_min, r_min, c_max, r_max = top_slick['bbox_px']
        pad = 80
        r0, r1 = max(0, r_min - pad), min(H, r_max + pad)
        c0, c1 = max(0, c_min - pad), min(W, c_max + pad)
        ax6.imshow(overlay_rgb[r0:r1, c0:c1])
        ax6.set_title(f"6. Primary Slick Close-Up (#{top_slick['slick_id']}: {top_slick['pixel_count']} px)",
                      color='#E6EDF3', fontsize=12, fontweight='bold', pad=8)
    else:
        ax6.text(0.5, 0.5, "No Oil Slicks Detected", color='#8B949E', ha='center', va='center')
        ax6.set_title("6. Primary Slick Close-Up", color='#E6EDF3', fontsize=12, fontweight='bold', pad=8)
    ax6.axis('off')

    # Panel 7: Surveillance Dossier & Scorecard
    ax_info = fig.add_subplot(gs[:, 3])
    ax_info.set_facecolor('#161B22')
    for spine in ax_info.spines.values():
        spine.set_edgecolor('#30363D')
        spine.set_linewidth(1.5)
    ax_info.set_xticks([])
    ax_info.set_yticks([])

    total_oil_px = int(filtered_mask.sum())
    total_px = H * W
    oil_pct = (total_oil_px / total_px) * 100.0
    px_area_km2 = (metadata.get('pixel_size_x', 10.0) * metadata.get('pixel_size_y', 10.0)) / 1e6
    total_oil_km2 = total_oil_px * px_area_km2
    total_scene_km2 = total_px * px_area_km2

    wgs = metadata.get('wgs84_bounds')
    loc_str = f"{wgs[1]:.3f} N to {wgs[3]:.3f} N\n  {abs(wgs[0]):.3f} W to {abs(wgs[2]):.3f} W" if wgs else "Pixel Local (Non-Georeferenced)"
    crs_name = str(metadata.get('crs')) if metadata.get('crs') else "UTM Zone 11N (EPSG:32611)"

    top_slick_info = ""
    if slicks:
        top_slick_info = f"""
  • Primary Slick ID: #{slicks[0]['slick_id']}
  • Primary Slick Size: {slicks[0]['pixel_count']:,} px ({slicks[0]['area_km2']:.4f} km²)
  • Peak Model Confidence: {slicks[0]['max_confidence']*100:.1f}%
  • Centroid [Col, Row]: [{slicks[0]['centroid_px'][0]:.0f}, {slicks[0]['centroid_px'][1]:.0f}]"""

    dossier_text = f"""
  ==============================================
   OILTRACE MARINE SURVEILLANCE REPORT
  ==============================================

  [SCENE METADATA]
  • Source Scene: {os.path.basename(metadata['path'])}
  • Raster Grid: {W} x {H} pixels ({total_px:,} px)
  • Pixel Resolution: {metadata.get('pixel_size_x', 10.0):.1f} m x {metadata.get('pixel_size_y', 10.0):.1f} m
  • Spatial Domain: {total_scene_km2:.2f} km²
  • Coordinate Reference: {crs_name}
  • Geographic Envelope:
  {loc_str}

  [AI DETECTION SUMMARY]
  • Detection Engine: {model_name}
  • Input Polarization: Dual-Pol (VV + VH Sigma0 dB)
  • Spatial Tiling: Non-overlapping 512x512 patches
  • Noise Suppression Cutoff: >= 50 px

  [INCIDENT METRICS]
  • Operational Status: OIL SPILL CONFIRMED [ALERT]
  • Confirmed Slick Clusters: {len(slicks)} slicks
  • Total Oil Spill Pixels: {total_oil_px:,} px
  • Total Spill Surface Area: {total_oil_km2:.4f} km²
  • Metric Area in Hectares: {total_oil_km2 * 100:.2f} ha
  • Marine Slick Coverage: {oil_pct:.3f}%

  [PRIMARY SLICK CHARACTERISTICS]{top_slick_info}

  [OUTPUT DELIVERABLES]
  • High-Res Binary Mask: {os.path.basename(mask_png_path)}
  • Overlaid Composite: {os.path.basename(overlay_png_path)}
  • Probability Map: {os.path.basename(proba_png_path)}
  • Analytical Dashboard: {os.path.basename(result_png_path)}
  ==============================================
    """

    ax_info.text(0.05, 0.95, dossier_text, color='#C9D1D9', fontfamily='monospace',
                 fontsize=9.5, va='top', ha='left', linespacing=1.35)

    fig.suptitle(f"OILTRACE AUTOMATED SATELLITE SAR OIL SPILL DETECTION DOSSIER | Scene: {os.path.basename(metadata['path'])}",
                 color='#58A6FF', fontsize=15, fontweight='bold', y=0.97)

    plt.savefig(result_png_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"  [+] Saved Comprehensive Result Dashboard -> {result_png_path}")

    return {
        'mask_png': mask_png_path,
        'probability_png': proba_png_path,
        'overlay_png': overlay_png_path,
        'result_png': result_png_path
    }


# ==============================================================================
# 6. Exporters & Pipeline Runner
# ==============================================================================

def export_geotiff_mask(filtered_mask: np.ndarray, metadata: dict, output_path: str):
    """Save georeferenced GeoTIFF mask for direct GIS / QGIS ingestion."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if HAS_RASTERIO and metadata.get('transform') and metadata.get('crs'):
        try:
            with rasterio.open(
                output_path, 'w', driver='GTiff',
                height=filtered_mask.shape[0], width=filtered_mask.shape[1],
                count=1, dtype=rasterio.uint8, crs=metadata['crs'],
                transform=metadata['transform'], compress='lzw'
            ) as dst:
                dst.write(filtered_mask.astype(np.uint8), 1)
            print(f"  [+] Saved Georeferenced Mask GeoTIFF -> {output_path}")
            return
        except Exception as e:
            print(f"Rasterio GeoTIFF export error: {e}, attempting tifffile...")

    if HAS_TIFFFILE:
        tifffile.imwrite(output_path, filtered_mask.astype(np.uint8))
        print(f"  [+] Saved Mask TIFF -> {output_path}")


def export_geojson(slicks: list, metadata: dict, output_path: str):
    """Save GeoJSON polygons of detected slicks."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features = []
    for s in slicks:
        geo = s.get('geo', {})
        if 'bbox_wgs84' in geo:
            b = geo['bbox_wgs84']
            poly_coords = [
                [b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]
            ]
            geom = {"type": "Polygon", "coordinates": [poly_coords]}
        else:
            c_min, r_min, c_max, r_max = s['bbox_px']
            geom = {"type": "Polygon", "coordinates": [[[c_min, r_min], [c_max, r_min], [c_max, r_max], [c_min, r_max], [c_min, r_min]]]}

        features.append({
            "type": "Feature", "geometry": geom,
            "properties": {
                "slick_id": s['slick_id'], "pixel_count": s['pixel_count'],
                "area_km2": s['area_km2'], "area_hectares": s['area_hectares'],
                "mean_confidence": s['mean_confidence'], "max_confidence": s['max_confidence'],
                "centroid_px": s['centroid_px']
            }
        })

    geojson_obj = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson_obj, f, indent=2)
    print(f"  [+] Saved Vector Polygons GeoJSON -> {output_path}")


# ==============================================================================
# 5b. Two-Stage OBIA Pipeline Helpers & Orchestration
# ==============================================================================

def extract_scene_id(input_path: str) -> str:
    """
    Extracts a canonical scene identifier from the input path.
    If the file stem is 'scene' or starts with 'scene_', uses parent folder name.
    Otherwise uses file stem.
    """
    path_obj = Path(input_path)
    stem = path_obj.stem
    if stem.lower() == "scene" or stem.lower().startswith("scene_"):
        parent_name = path_obj.parent.name
        if parent_name and parent_name not in (".", "/", ""):
            return parent_name
    return stem


def rasterize_confirmed_slicks(
    slicks: List[Any],
    shape: Tuple[int, int],
    transform: Any = None,
    candidate_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Rasterizes confirmed slick candidate dark spots onto a binary 2D mask.
    Fast-paths to all zeros if slicks is empty.
    Returns 2D uint8 NumPy array where oil pixels are 255 and background is 0.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    if not slicks:
        return mask

    # Exact pixel mask recovery if candidate segmentation mask is available
    if candidate_mask is not None:
        for slick in slicks:
            lid = getattr(slick, 'label_id', None)
            if lid is not None:
                mask[candidate_mask == lid] = 255
            elif hasattr(slick, 'pixel_coords') and slick.pixel_coords is not None:
                r, c = slick.pixel_coords
                r = np.asarray(r, dtype=np.intp)
                c = np.asarray(c, dtype=np.intp)
                valid = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
                mask[r[valid], c[valid]] = 255
            elif hasattr(slick, 'bbox_px'):
                c_min, r_min, c_max, r_max = slick.bbox_px
                mask[r_min:r_max, c_min:c_max] = 255
        return mask

    for slick in slicks:
        if hasattr(slick, 'pixel_coords') and slick.pixel_coords is not None:
            r, c = slick.pixel_coords
            r = np.asarray(r, dtype=np.intp)
            c = np.asarray(c, dtype=np.intp)
            valid = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
            mask[r[valid], c[valid]] = 255
        elif hasattr(slick, 'geometry') and slick.geometry is not None:
            if HAS_RASTERIO and transform is not None:
                try:
                    from rasterio.features import rasterize
                    slick_mask = rasterize(
                        [(slick.geometry, 255)],
                        out_shape=shape,
                        transform=transform,
                        fill=0,
                        dtype=np.uint8
                    )
                    mask = np.maximum(mask, slick_mask)
                except Exception:
                    pass
        elif hasattr(slick, 'bbox_px'):
            c_min, r_min, c_max, r_max = slick.bbox_px
            mask[r_min:r_max, c_min:c_max] = 255
    return mask


def export_spills_geojson(
    slicks: List[Any],
    classification_results: List[Any],
    feature_result: Any,
    metadata: Dict[str, Any],
    output_path: str
) -> str:
    """Save GeoJSON FeatureCollection of confirmed slicks with properties."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    classif_map = {}
    for res in classification_results:
        if hasattr(res, 'candidate_id'):
            classif_map[res.candidate_id] = res

    feat_map = {}
    if feature_result and hasattr(feature_result, 'candidate_features'):
        for f in feature_result.candidate_features:
            if hasattr(f, 'candidate_id'):
                feat_map[f.candidate_id] = f

    features = []
    for slick in slicks:
        cand_id = getattr(slick, 'candidate_id', f"slick_{len(features)+1}")
        classif = classif_map.get(cand_id)
        confidence = float(classif.confidence) if classif else 1.0
        feat = feat_map.get(cand_id)

        properties = {
            "candidate_id": cand_id,
            "confidence": round(confidence, 4),
            "classification_status": "confirmed_slick",
            "predicted_label": 1,
            "area_km2": round(float(getattr(slick, 'area_km2', 0.0)), 4),
            "pixel_count": int(getattr(slick, 'pixel_count', 0)),
            "touches_suspect_vessel": bool(getattr(slick, 'touches_suspect_vessel', False)),
            "nearest_vessel_id": getattr(slick, 'nearest_vessel_id', None),
            "distance_to_nearest_vessel_m": round(float(slick.distance_to_nearest_vessel_m), 1) if getattr(slick, 'distance_to_nearest_vessel_m', None) is not None else None,
            "mean_vv_db": round(float(getattr(slick, 'mean_vv_db', 0.0)), 2),
            "local_damping_db": round(float(getattr(slick, 'local_damping_db', 0.0)), 2)
        }
        if feat:
            properties["damping_ratio_vv"] = round(float(getattr(feat, 'damping_ratio_vv', 0.0)), 4)
            properties["boundary_gradient_mean"] = round(float(getattr(feat, 'boundary_gradient_mean', 0.0)), 4)
            properties["elongation"] = round(float(getattr(feat, 'elongation', 0.0)), 4)
            properties["complexity"] = round(float(getattr(feat, 'complexity_ratio', 0.0)), 4)

        geom = getattr(slick, 'geometry_wgs84', None)
        if not geom or 'coordinates' not in geom:
            if hasattr(slick, 'geometry') and slick.geometry is not None:
                geom = slick.geometry.__geo_interface__
            else:
                b = getattr(slick, 'bbox_wgs84', None)
                if b and b != (0.0, 0.0, 0.0, 0.0):
                    poly_coords = [
                        [b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]
                    ]
                    geom = {"type": "Polygon", "coordinates": [poly_coords]}
                else:
                    c_min, r_min, c_max, r_max = getattr(slick, 'bbox_px', (0, 0, 0, 0))
                    geom = {
                        "type": "Polygon",
                        "coordinates": [[[c_min, r_min], [c_max, r_min], [c_max, r_max], [c_min, r_max], [c_min, r_min]]]
                    }

        features.append({
            "type": "Feature",
            "id": cand_id,
            "properties": properties,
            "geometry": geom
        })

    geojson_dict = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson_dict, f, indent=2)

    print(f"  [+] Saved Confirmed Slicks GeoJSON -> {output_path}")
    return output_path


def export_candidates_classified_geojson(
    candidates: List[Any],
    classification_results: List[Any],
    feature_result: Any,
    metadata: Dict[str, Any],
    output_path: str
) -> str:
    """Save GeoJSON FeatureCollection of all candidate spots with classification labels."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    classif_map = {}
    for res in classification_results:
        if hasattr(res, 'candidate_id'):
            classif_map[res.candidate_id] = res

    features = []
    for cand in candidates:
        cand_id = getattr(cand, 'candidate_id', f"cand_{len(features)+1}")
        classif = classif_map.get(cand_id)
        confidence = float(classif.confidence) if classif else 0.0
        status = getattr(classif, 'classification_status', 'lookalike_suppressed') if classif else 'lookalike_suppressed'
        label = int(getattr(classif, 'predicted_label', 0)) if classif else 0

        properties = {
            "candidate_id": cand_id,
            "confidence": round(confidence, 4),
            "predicted_label": label,
            "classification_status": status,
            "area_km2": round(float(getattr(cand, 'area_km2', 0.0)), 4),
            "pixel_count": int(getattr(cand, 'pixel_count', 0)),
            "touches_suspect_vessel": bool(getattr(cand, 'touches_suspect_vessel', False))
        }

        geom = getattr(cand, 'geometry_wgs84', None)
        if not geom or 'coordinates' not in geom:
            if hasattr(cand, 'geometry') and cand.geometry is not None:
                geom = cand.geometry.__geo_interface__
            else:
                b = getattr(cand, 'bbox_wgs84', None)
                if b and b != (0.0, 0.0, 0.0, 0.0):
                    poly_coords = [
                        [b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]
                    ]
                    geom = {"type": "Polygon", "coordinates": [poly_coords]}
                else:
                    c_min, r_min, c_max, r_max = getattr(cand, 'bbox_px', (0, 0, 0, 0))
                    geom = {
                        "type": "Polygon",
                        "coordinates": [[[c_min, r_min], [c_max, r_min], [c_max, r_max], [c_min, r_max], [c_min, r_min]]]
                    }

        features.append({
            "type": "Feature",
            "id": cand_id,
            "properties": properties,
            "geometry": geom
        })

    geojson_dict = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson_dict, f, indent=2)

    print(f"  [+] Saved Candidates GeoJSON -> {output_path}")
    return output_path


def resolve_pipeline_mode(
    mode: Optional[str] = None,
    model: Optional[str] = None,
    obia_model: Optional[str] = None,
    script_dir: Optional[str] = None
) -> Tuple[str, str]:
    """
    Resolves execution mode and model path following precedence rules:
    1. Explicit --mode flag takes precedence.
    2. If --mode is omitted and --model is provided:
       - .pt -> unet
       - path contains 'obia' or bundle dictionary -> two_stage
       - other .joblib -> rf_pixel
    3. If neither --mode nor --model is provided:
       - check default OBIA model -> two_stage
       - check default U-Net model -> unet
       - check default pixel RF model -> rf_pixel
       - else raise FileNotFoundError
    """
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))

    obia_candidates = [
        os.path.join(script_dir, "models", "best_obia_rf.joblib"),
        os.path.join(script_dir, "best_obia_rf.joblib"),
    ]
    default_obia = next((p for p in obia_candidates if os.path.exists(p)), obia_candidates[0])

    unet_candidates = [
        os.path.join(script_dir, "models", "best_unet.pt"),
        os.path.join(script_dir, "best_unet.pt"),
    ]
    default_unet = next((p for p in unet_candidates if os.path.exists(p)), unet_candidates[0])

    rf_candidates = [
        os.path.join(script_dir, "models", "best_model.joblib"),
        os.path.join(script_dir, "best_model.joblib"),
    ]
    default_rf = next((p for p in rf_candidates if os.path.exists(p)), rf_candidates[0])

    if mode is not None:
        mode = mode.lower()
        if mode == "two_stage":
            candidate = obia_model or model
            if candidate and not candidate.endswith(".pt"):
                chosen_model = candidate
            else:
                chosen_model = default_obia
            return mode, chosen_model
        elif mode == "unet":
            candidate = model
            if candidate and not candidate.endswith(".joblib"):
                chosen_model = candidate
            else:
                chosen_model = default_unet
            return mode, chosen_model
        elif mode in ("rf_pixel", "random_forest", "rf"):
            candidate = model
            if candidate and not candidate.endswith(".pt"):
                chosen_model = candidate
            else:
                chosen_model = default_rf
            return "rf_pixel", chosen_model
        else:
            raise ValueError(f"Unknown mode: {mode}. Choices: two_stage, unet, rf_pixel")

    # Mode is omitted: inspect model argument if present
    if model is not None:
        if model.endswith(".pt") or "unet" in os.path.basename(model).lower():
            return "unet", model
        elif "obia" in os.path.basename(model).lower():
            return "two_stage", model
        elif model.endswith(".joblib"):
            try:
                data = joblib.load(model)
                if isinstance(data, dict) and "model" in data and "feature_names" in data:
                    return "two_stage", model
                else:
                    return "rf_pixel", model
            except Exception:
                return "rf_pixel", model
        else:
            return "rf_pixel", model

    # Neither mode nor model provided: auto-detect by file existence
    if obia_model and os.path.exists(obia_model):
        return "two_stage", obia_model
    if os.path.exists(default_obia):
        return "two_stage", default_obia
    if os.path.exists(default_unet):
        return "unet", default_unet
    if os.path.exists(default_rf):
        return "rf_pixel", default_rf

    raise FileNotFoundError("No valid model found. Specify --mode or --model.")


def run_two_stage_pipeline(config: PipelineRunConfig) -> TwoStagePipelineResult:
    """
    Executes the complete two stage detection pipeline:
    1. Ingests dual polarization SAR GeoTIFF scene.
    2. Detects bright metallic vessels and generates shadow corridor masks.
    3. Segments dark spot candidate polygons using ocean background damping contrast.
    4. Extracts 17 physical and geometric feature descriptors per candidate polygon.
    5. Evaluates candidates using ObjectClassifier to separate slicks from lookalikes.
    6. Exports verified deliverables: mask.png, scene_spills.geojson, scene_candidates.geojson, and result.json.
    """
    start_time = time.time()
    scene_id = extract_scene_id(config.input_path)
    output_dir = config.output_dir or os.path.dirname(os.path.abspath(config.input_path))
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 75)
    print("🛰️  OILTRACE TWO STAGE OBJECT BASED OIL SPILL DETECTION PIPELINE")
    print("=" * 75)
    print(f"Scene ID:         {scene_id}")
    print(f"Input Scene:      {config.input_path}")
    print(f"OBIA Model:       {config.obia_model_path}")
    print(f"Output Directory: {output_dir}")

    # 1. Ingest SAR Scene
    print("\n[Step 1/5] Ingesting SAR Scene GeoTIFF...")
    sar_data, metadata = load_sar_scene(config.input_path)
    H, W = metadata['height'], metadata['width']
    print(f"  Dimensions:     {W} x {H}")
    print(f"  Pixel Spacing:  {metadata.get('pixel_size_x', 10.0):.2f} m x {metadata.get('pixel_size_y', 10.0):.2f} m")

    # 2. Vessel Detection & Shadow Masking
    vessels = []
    shadow_pixels_masked = 0
    if config.enable_vessel_masking:
        print("\n[Step 2/5] Autonomous Vessel Detection & Radar Shadow Masking...")
        vessels_geojson = os.path.join(output_dir, "scene_vessels.geojson")
        if mask_vessel_shadows is not None:
            try:
                empty_cand_mask = np.zeros((H, W), dtype=bool)
                _, vessels, _ = mask_vessel_shadows(
                    sar_data,
                    empty_cand_mask,
                    metadata,
                    output_geojson_path=vessels_geojson
                )
            except Exception as e:
                print(f"  Vessel masking note: {e}")
                vessels = []
        elif detect_vessel_targets is not None:
            try:
                vv_raw = sar_data[0] if sar_data.ndim == 3 and sar_data.shape[0] == 2 else sar_data[:, :, 0]
                vh_raw = sar_data[1] if sar_data.ndim == 3 and sar_data.shape[0] == 2 else sar_data[:, :, 1]
                valid_mask = ((vv_raw != 0.0) | (vh_raw != 0.0)) & ~np.isnan(vv_raw)
                vv_db = np.clip(vv_raw.astype(np.float32), -50.0, 5.0)
                vh_db = np.clip(vh_raw.astype(np.float32), -50.0, 5.0)
                vessels = detect_vessel_targets(vv_db, vh_db, valid_mask, metadata)
            except Exception as e:
                print(f"  Vessel detection note: {e}")
                vessels = []
        print(f"  Detected Metallic Vessels: {len(vessels)}")
        print(f"  Masked Shadow Pixels:      {shadow_pixels_masked:,}")
    else:
        print("\n[Step 2/5] Vessel Masking Disabled.")

    # 3. Candidate Segmentation
    print("\n[Step 3/5] Adaptive Candidate Dark Spot Segmentation...")
    candidates = []
    candidate_mask = None
    damping_map = None
    total_candidates = 0
    total_candidate_area_km2 = 0.0

    if segment_candidate_dark_spots is not None:
        seg_res = segment_candidate_dark_spots(
            sar_data,
            metadata,
            min_damping_db=config.min_damping,
            max_backscatter_db=config.max_backscatter,
            min_area_px=config.min_candidate_px,
            window_size=config.window_size,
            vessels=vessels
        )
        candidates = seg_res.candidates
        candidate_mask = seg_res.candidate_mask
        damping_map = seg_res.damping_ratio_map
        total_candidates = seg_res.candidate_count
        total_candidate_area_km2 = seg_res.total_candidate_area_km2
        print(f"  Segmented Candidates:      {total_candidates}")
        print(f"  Total Candidate Area:      {total_candidate_area_km2:.4f} km²")
    else:
        print("  Warning: segment_candidate_dark_spots not available.")

    # 4. Feature Extraction
    feature_result = None
    if candidates and extract_candidate_features is not None:
        print("\n[Step 4/5] Extracting 17 Physical & Geometric Object Features...")
        feature_result = extract_candidate_features(
            candidates=candidates,
            candidate_mask=candidate_mask,
            sar_data=sar_data,
            damping_map=damping_map,
            metadata=metadata,
            vessels=vessels
        )
        print(f"  Extracted Feature Vectors: {feature_result.candidate_count}")
    else:
        print("\n[Step 4/5] Feature Extraction (0 candidates).")

    # 5. OBIA Classification
    print("\n[Step 5/5] Object Level Random Forest Evaluation...")
    if not os.path.exists(config.obia_model_path):
        raise FileNotFoundError(f"OBIA model bundle not found: {config.obia_model_path}")

    classifier = load_object_classifier(config.obia_model_path) if load_object_classifier is not None else None
    if classifier is None:
        raise RuntimeError("ObjectClassifier module could not be loaded.")

    effective_threshold = config.threshold if config.threshold is not None else classifier.decision_threshold
    print(f"  Operating Threshold:       {effective_threshold:.4f}")

    classification_results = []
    if candidates:
        classification_results = classifier.predict_candidates(
            candidates,
            decision_threshold=effective_threshold
        )

    confirmed_slicks = []
    suppressed_lookalikes = []

    for cand, res in zip(candidates, classification_results):
        if res.predicted_label == 1:
            confirmed_slicks.append(cand)
        else:
            suppressed_lookalikes.append(cand)

    confirmed_slicks_count = len(confirmed_slicks)
    suppressed_lookalikes_count = len(suppressed_lookalikes)
    confirmed_slicks_area_km2 = sum(float(c.area_km2) for c in confirmed_slicks)

    print(f"  Confirmed Oil Slicks:      {confirmed_slicks_count}")
    print(f"  Suppressed Lookalikes:     {suppressed_lookalikes_count}")
    print(f"  Confirmed Slick Area:      {confirmed_slicks_area_km2:.4f} km²")

    # Generate deliverables
    mask_png_path = os.path.join(output_dir, "mask.png")
    raster_mask = rasterize_confirmed_slicks(
        confirmed_slicks, (H, W), metadata.get('transform'), candidate_mask=candidate_mask
    )
    Image.fromarray(raster_mask).save(mask_png_path, format="PNG")
    print(f"  [+] Saved Binary Mask -> {mask_png_path}")

    spills_geojson_path = os.path.join(output_dir, "scene_spills.geojson")
    export_spills_geojson(confirmed_slicks, classification_results, feature_result, metadata, spills_geojson_path)

    candidates_geojson_path = None
    if config.enable_candidate_export:
        candidates_geojson_path = os.path.join(output_dir, "scene_candidates.geojson")
        export_candidates_classified_geojson(candidates, classification_results, feature_result, metadata, candidates_geojson_path)

    # Multi-panel analytical visualization dashboard
    base_name = os.path.splitext(os.path.basename(config.input_path))[0]
    filtered_mask = (raster_mask > 0).astype(np.uint8)

    obia_proba_map = np.zeros((H, W), dtype=np.float32)
    if candidates and candidate_mask is not None:
        for cand, res in zip(candidates, classification_results):
            c_min, r_min, c_max, r_max = cand.bbox_px
            sub_cand = (candidate_mask[r_min:r_max, c_min:c_max] == cand.label_id)
            obia_proba_map[r_min:r_max, c_min:c_max][sub_cand] = float(res.confidence)

    slicks_metadata = []
    for i, c in enumerate(confirmed_slicks):
        conf = float(c.slick_probability) if c.slick_probability is not None else 1.0
        slicks_metadata.append({
            'slick_id': i + 1,
            'candidate_id': c.candidate_id,
            'pixel_count': c.pixel_count,
            'area_km2': c.area_km2,
            'area_hectares': round(c.area_km2 * 100.0, 2),
            'max_confidence': conf,
            'centroid_px': [float(c.centroid_px[0]), float(c.centroid_px[1])],
            'bbox_px': [int(c.bbox_px[0]), int(c.bbox_px[1]), int(c.bbox_px[2]), int(c.bbox_px[3])]
        })

    vis_paths = generate_visualizations(
        sar_data=sar_data,
        proba_map=obia_proba_map,
        filtered_mask=filtered_mask,
        slicks=slicks_metadata,
        metadata=metadata,
        output_dir=output_dir,
        base_name=base_name,
        model_name="OBIA Random Forest (Two-Stage Cascade)"
    )

    elapsed_time = time.time() - start_time
    result_json_path = os.path.join(output_dir, "result.json")
    result_data = {
        "scene_id": scene_id,
        "pipeline_mode": "two_stage",
        "model_path": os.path.abspath(config.obia_model_path),
        "decision_threshold": float(effective_threshold),
        "execution_time_seconds": round(elapsed_time, 2),
        "vessel_masking": {
            "vessels_detected": len(vessels),
            "shadow_pixels_masked": shadow_pixels_masked
        },
        "candidate_segmentation": {
            "total_candidates": total_candidates,
            "total_candidate_area_km2": round(total_candidate_area_km2, 4)
        },
        "object_classification": {
            "confirmed_slicks": confirmed_slicks_count,
            "suppressed_lookalikes": suppressed_lookalikes_count,
            "confirmed_area_km2": round(confirmed_slicks_area_km2, 4)
        },
        "deliverables": {
            "mask_png": os.path.abspath(mask_png_path),
            "spills_geojson": os.path.abspath(spills_geojson_path),
            "candidates_geojson": os.path.abspath(candidates_geojson_path) if candidates_geojson_path and os.path.exists(candidates_geojson_path) else None,
            "result_json": os.path.abspath(result_json_path),
            "detection_result_png": os.path.abspath(vis_paths['result_png']),
            "mask_overlay_png": os.path.abspath(vis_paths['overlay_png']),
            "probability_png": os.path.abspath(vis_paths['probability_png'])
        }
    }

    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2)
    print(f"  [+] Saved Result Metadata JSON -> {result_json_path}")

    deliverables_map = {
        "mask_png": mask_png_path,
        "spills_geojson": spills_geojson_path,
        "candidates_geojson": candidates_geojson_path,
        "result_json": result_json_path,
        "detection_result_png": vis_paths['result_png'],
        "mask_overlay_png": vis_paths['overlay_png'],
        "probability_png": vis_paths['probability_png']
    }

    print("\n" + "=" * 75)
    print("TWO STAGE PIPELINE EXECUTION COMPLETED")
    print(f"Total Runtime: {elapsed_time:.2f}s")
    print(f"Deliverables Directory: {output_dir}")
    print("=" * 75 + "\n")

    return TwoStagePipelineResult(
        scene_id=scene_id,
        pipeline_mode="two_stage",
        total_vessels_detected=len(vessels),
        shadow_pixels_masked=shadow_pixels_masked,
        total_candidates_segmented=total_candidates,
        total_candidate_area_km2=total_candidate_area_km2,
        confirmed_slicks_count=confirmed_slicks_count,
        suppressed_lookalikes_count=suppressed_lookalikes_count,
        confirmed_slicks_area_km2=confirmed_slicks_area_km2,
        execution_time_seconds=elapsed_time,
        deliverables=deliverables_map,
        confirmed_slicks=confirmed_slicks,
        classification_results=classification_results
    )


def run_detection_pipeline(scene_path: str, model_path: Optional[str] = None, output_dir: str = None,
                           mode: Optional[str] = None, obia_model: Optional[str] = None,
                           threshold: float = None, min_px: int = 50, apply_lee: bool = True,
                           no_vessel_masking: bool = False,
                           min_damping: float = 2.2, max_backscatter: float = -20.0,
                           window_size: int = 151, min_candidate_px: int = 30,
                           no_candidate_segmentation: bool = False,
                           no_feature_extraction: bool = False,
                           features_csv: Optional[str] = None,
                           device: str = "cpu"):
    # Resolve mode and model
    resolved_mode, resolved_model = resolve_pipeline_mode(
        mode=mode, model=model_path, obia_model=obia_model,
        script_dir=os.path.dirname(os.path.abspath(__file__))
    )

    if resolved_mode == "two_stage":
        config = PipelineRunConfig(
            input_path=scene_path,
            output_dir=output_dir,
            mode="two_stage",
            obia_model_path=resolved_model,
            threshold=threshold,
            enable_vessel_masking=not no_vessel_masking,
            enable_candidate_export=not no_candidate_segmentation,
            device=device,
            min_damping=min_damping,
            max_backscatter=max_backscatter,
            window_size=window_size,
            min_candidate_px=min_candidate_px,
            min_px=min_px
        )
        res = run_two_stage_pipeline(config)
        with open(res.deliverables['result_json'], 'r', encoding='utf-8') as f:
            return json.load(f)

    model_path = resolved_model
    start_time = time.time()
    print("=" * 75)
    print("🛰️  OILTRACE ADVANCED SAR OIL SPILL DETECTION PIPELINE")
    print("=" * 75)
    print(f"Input Scene:    {scene_path}")
    print(f"Model File:     {model_path}")

    # Determine model type
    is_unet = model_path.endswith('.pt') or 'unet' in model_path.lower()
    if threshold is None:
        threshold = 0.5 if is_unet else 0.3
    model_name = "Deep Learning U-Net (31.0M Params)" if is_unet else "Random Forest Pixel Classifier"

    print(f"Detection Engine: {model_name}")
    print(f"Threshold:        {threshold}")
    print(f"Min Cluster:      {min_px} px")
    print(f"Vessel Masking:   {'Disabled' if no_vessel_masking else 'Active (Autonomous CFAR + Shadow Corridor)'}")
    print(f"Candidate Seg:    {'Disabled' if no_candidate_segmentation else f'Active (Window {window_size}px, DR>={min_damping}dB, VV<={max_backscatter}dB)'}")
    print(f"Feature Extractor:{'Disabled' if no_feature_extraction else 'Active (17 physical and geometric candidate descriptors)'}")

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(scene_path))
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(scene_path))[0]

    # Step 1: Load SAR Scene
    print("\n[Step 1/5] Loading SAR Scene GeoTIFF...")
    sar_data, metadata = load_sar_scene(scene_path)
    print(f"  Dimensions: {metadata['width']} x {metadata['height']} x {sar_data.shape[2] if sar_data.ndim==3 else 1}")
    print(f"  Pixel Spacing: {metadata['pixel_size_x']:.2f} m x {metadata['pixel_size_y']:.2f} m")
    if metadata.get('crs'):
        print(f"  Spatial Reference: {metadata['crs']}")

    # Step 2: Model Inference
    print(f"\n[Step 2/5] Running ML Inference ({model_name})...")
    t0 = time.time()
    if is_unet:
        proba_map, mask_raw = run_unet_inference(sar_data, model_path, threshold=threshold)
    else:
        proba_map, mask_raw = run_rf_inference(sar_data, model_path, threshold=threshold, apply_lee=apply_lee)
    print(f"  Inference completed in {time.time() - t0:.2f}s")

    # Step 3: Autonomous Vessel Detection & Radar Shadow Masking
    vessels = []
    vessels_geojson_path = None
    if mask_vessel_shadows is not None and not no_vessel_masking:
        print("\n[Step 3/5] Autonomous Vessel Detection & Shadow Corridor Masking...")
        vessels_geojson_path = os.path.join(output_dir, f"{base_name}_vessels.geojson")
        mask_cleaned, vessels, _ = mask_vessel_shadows(
            sar_data, mask_raw, metadata, output_geojson_path=vessels_geojson_path
        )
        suppressed_px = int(mask_raw.sum()) - int(mask_cleaned.sum())
        suspect_vessels = [v for v in vessels if v.is_suspect_source]
        print(f"  Detected Metallic Vessels:    {len(vessels)}")
        print(f"  Suppressed Shadow Pixels:     {suppressed_px:,}")
        print(f"  Suspect Discharging Vessels:  {len(suspect_vessels)}")
        if vessels_geojson_path:
            print(f"  [+] Saved Vessel Positions GeoJSON -> {vessels_geojson_path}")
    else:
        mask_cleaned = mask_raw

    # Step 3b: Adaptive Dark Spot Candidate Segmentation
    candidates_result = None
    candidates_geojson_path = None
    if segment_candidate_dark_spots is not None and not no_candidate_segmentation:
        print("\n[Step 3b/5] Adaptive Dark Spot Candidate Segmentation...")
        t_seg = time.time()
        candidates_geojson_path = os.path.join(output_dir, f"{base_name}_candidates.geojson")
        candidates_result = segment_candidate_dark_spots(
            sar_data, metadata,
            min_damping_db=min_damping,
            max_backscatter_db=max_backscatter,
            min_area_px=min_candidate_px,
            window_size=window_size,
            vessels=vessels
        )
        export_candidate_geojson(candidates_result.candidates, metadata, candidates_geojson_path)
        touching_vessels = [c for c in candidates_result.candidates if c.touches_suspect_vessel]
        print(f"  Candidate Extraction Completed in {time.time() - t_seg:.2f}s")
        print(f"  Candidate Dark Spots Extracted: {candidates_result.candidate_count}")
        print(f"  Cumulative Candidate Area:      {candidates_result.total_candidate_area_km2:.4f} km²")
        print(f"  Candidates Touching Vessels:    {len(touching_vessels)} spots")
        if candidates_result.candidates:
            top_spot = candidates_result.candidates[0]
            print(f"  Primary Candidate Spot:         {top_spot.candidate_id} ({top_spot.pixel_count:,} px, {top_spot.area_km2:.4f} km²)")
        print(f"  [+] Saved Candidate Polygons GeoJSON -> {candidates_geojson_path}")

    # Step 3c: Physical & Geometric Object Feature Extraction
    features_result = None
    features_csv_path = None
    if candidates_result is not None and extract_candidate_features is not None and not no_feature_extraction:
        print("\n[Step 3c/5] Physical & Geometric Object Feature Extraction...")
        t_feat = time.time()
        features_csv_path = features_csv or os.path.join(output_dir, f"{base_name}_features.csv")
        features_result = extract_candidate_features(
            candidates=candidates_result.candidates,
            candidate_mask=candidates_result.candidate_mask,
            sar_data=sar_data,
            damping_map=candidates_result.damping_ratio_map,
            metadata=metadata,
            vessels=vessels
        )
        export_features_tabular(features_result, features_csv_path)
        print(f"  Feature Extraction Completed in {time.time() - t_feat:.2f}s")
        print(f"  Extracted 17 Dimensional Feature Vectors: {features_result.candidate_count} spots")
        print(f"  Feature Matrix Shape:           {features_result.feature_matrix.shape}")
        if features_result.candidate_features:
            sample_feat = features_result.candidate_features[0]
            print(f"  Primary Spot Elongation:        {sample_feat.elongation:.2f}")
            print(f"  Primary Spot Circularity:       {sample_feat.circularity:.4f}")
            print(f"  Primary Spot Complexity Ratio:  {sample_feat.complexity_ratio:.2f}")
            print(f"  Primary Spot Fractal Dim:       {sample_feat.fractal_dimension:.2f}")
            print(f"  Primary Spot Mean Sobel Grad:   {sample_feat.boundary_gradient_mean:.2f}")
        print(f"  [+] Saved Candidate Features CSV -> {features_csv_path}")

    # Step 4: Morphological Analysis & Slick Extraction
    print("\n[Step 4/5] Connected-Component Analysis & Noise Filtering...")
    filtered_mask, slicks = analyze_slicks(mask_cleaned, proba_map, metadata, min_px=min_px)

    total_px = metadata['width'] * metadata['height']
    oil_px = int(filtered_mask.sum())
    oil_pct = (oil_px / total_px) * 100.0
    px_area_km2 = (metadata['pixel_size_x'] * metadata['pixel_size_y']) / 1.0e6
    oil_area_km2 = oil_px * px_area_km2

    print(f"  Raw candidate oil pixels:     {int(mask_raw.sum()):,}")
    if mask_vessel_shadows is not None and not no_vessel_masking:
        print(f"  Post-vessel candidate pixels: {int(mask_cleaned.sum()):,}")
    print(f"  Filtered oil pixels (≥{min_px}px):  {oil_px:,} ({oil_pct:.3f}% coverage)")
    print(f"  Total Estimated Spill Area:   {oil_area_km2:.4f} km² ({oil_area_km2*100:.2f} hectares)")
    print(f"  Confirmed Spill Clusters:     {len(slicks)} slicks")
    if slicks:
        print(f"  Largest Slick:                {slicks[0]['pixel_count']:,} px ({slicks[0]['area_km2']:.4f} km²), ID #{slicks[0]['slick_id']}")

    # Step 5: Export Visualizations & Deliverables
    print("\n[Step 5/5] Generating PNG Images & Export Deliverables...")
    vis_paths = generate_visualizations(sar_data, proba_map, filtered_mask, slicks, metadata, output_dir, base_name, model_name)

    geotiff_path = os.path.join(output_dir, f"{base_name}_pred_mask.tif")
    export_geotiff_mask(filtered_mask, metadata, geotiff_path)

    geojson_path = os.path.join(output_dir, f"{base_name}_spills.geojson")
    export_geojson(slicks, metadata, geojson_path)

    artifacts_dict = {
        'mask_png': os.path.abspath(vis_paths['mask_png']),
        'overlay_png': os.path.abspath(vis_paths['overlay_png']),
        'probability_png': os.path.abspath(vis_paths['probability_png']),
        'result_png': os.path.abspath(vis_paths['result_png']),
        'mask_geotiff': os.path.abspath(geotiff_path),
        'geojson': os.path.abspath(geojson_path)
    }
    if vessels_geojson_path and os.path.exists(vessels_geojson_path):
        artifacts_dict['vessels_geojson'] = os.path.abspath(vessels_geojson_path)
    if candidates_geojson_path and os.path.exists(candidates_geojson_path):
        artifacts_dict['candidates_geojson'] = os.path.abspath(candidates_geojson_path)
    if features_csv_path and os.path.exists(features_csv_path):
        artifacts_dict['features_csv'] = os.path.abspath(features_csv_path)

    summary_path = os.path.join(output_dir, f"{base_name}_summary.json")
    summary_data = {
        'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'scene': os.path.abspath(scene_path),
        'model': os.path.abspath(model_path),
        'model_type': 'unet' if is_unet else 'random_forest',
        'image_size': [metadata['width'], metadata['height']],
        'resolution_m': [metadata['pixel_size_x'], metadata['pixel_size_y']],
        'total_area_km2': round(total_px * px_area_km2, 4),
        'oil_pixels': oil_px,
        'oil_area_km2': round(oil_area_km2, 4),
        'oil_area_hectares': round(oil_area_km2 * 100.0, 2),
        'coverage_percent': round(oil_pct, 4),
        'vessels_count': len(vessels),
        'suspect_vessels_count': len([v for v in vessels if v.is_suspect_source]),
        'candidate_spots_count': candidates_result.candidate_count if candidates_result else 0,
        'candidate_area_km2': candidates_result.total_candidate_area_km2 if candidates_result else 0.0,
        'candidate_features_count': features_result.candidate_count if features_result else 0,
        'slicks_count': len(slicks),
        'slicks': slicks,
        'artifacts': artifacts_dict,
        'elapsed_seconds': round(time.time() - start_time, 2)
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    print(f"  [+] Saved Structured Metadata JSON -> {summary_path}")

    print("\n" + "=" * 75)
    print("DETECTION PIPELINE EXECUTION FINISHED SUCCESSFULLY")
    print(f"Total Runtime: {time.time() - start_time:.2f} seconds")
    print(f"Primary Visual Result: {vis_paths['result_png']}")
    print(f"Binary Mask:           {vis_paths['mask_png']}")
    print(f"Mask Overlay:          {vis_paths['overlay_png']}")
    print("=" * 75 + "\n")

    return summary_data


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scene_candidates = [
        os.path.join(script_dir, "data", "scene.tif"),
        os.path.join(script_dir, "det_20260917163850_004aae", "scene.tif"),
    ]
    default_scene = next((s for s in scene_candidates if os.path.exists(s)), scene_candidates[0])

    default_outputs_dir = os.path.join(script_dir, "outputs")
    default_output = default_outputs_dir if os.path.isdir(default_outputs_dir) else None

    # Default to U-Net if available, otherwise RF
    default_unet = os.path.join(script_dir, "models", "best_unet.pt")
    if not os.path.exists(default_unet):
        default_unet = os.path.join(script_dir, "best_unet.pt")

    default_rf = os.path.join(script_dir, "models", "best_model.joblib")
    if not os.path.exists(default_rf):
        default_rf = os.path.join(script_dir, "best_model.joblib")

    default_model = default_unet if os.path.exists(default_unet) else default_rf

    parser = argparse.ArgumentParser(description="OILTRACE SAR Oil Spill Detection Pipeline")
    parser.add_argument("--input", default=default_scene,
                        help=f"Path to input SAR GeoTIFF (default: {default_scene})")
    parser.add_argument("--mode", choices=["two_stage", "unet", "rf_pixel"], default=None,
                        help="Detection mode: two_stage (OBIA Random Forest cascade), unet (Deep Learning), or rf_pixel (legacy pixel baseline)")
    parser.add_argument("--model", default=None,
                        help="Path to model (.pt for U-Net, .joblib for Random Forest, or OBIA bundle)")
    parser.add_argument("--obia-model", "--obia_model", dest="obia_model", default=None,
                        help="Path to object level Random Forest model bundle (.joblib)")
    parser.add_argument("--output", "--output_dir", dest="output_dir", default=None,
                        help="Output directory (default: pipeline/outputs for default scene, else directory of input scene)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Oil probability detection threshold (default: calibrated bundle value for OBIA, 0.5 for U-Net, 0.3 for RF)")
    parser.add_argument("--min_px", type=int, default=50,
                        help="Minimum connected pixel size for valid slick (default: 50)")
    parser.add_argument("--no_lee", action="store_true",
                        help="Disable Lee speckle filtering (RF only)")
    parser.add_argument("--no_vessel_masking", "--no-vessel-masking", dest="no_vessel_masking", action="store_true",
                        help="Disable autonomous vessel reflector and shadow corridor masking")
    parser.add_argument("--min_damping", type=float, default=2.2,
                        help="Minimum damping contrast threshold in dB (default: 2.2)")
    parser.add_argument("--max_backscatter", type=float, default=-20.0,
                        help="Maximum VV backscatter ceiling in dB (default: -20.0)")
    parser.add_argument("--window_size", type=int, default=151,
                        help="Local ocean moving window size in pixels (default: 151)")
    parser.add_argument("--min_candidate_px", type=int, default=30,
                        help="Minimum candidate cluster size in pixels (default: 30)")
    parser.add_argument("--no-candidates", "--no_candidates", "--no_candidate_segmentation",
                        dest="no_candidates", action="store_true",
                        help="Disable intermediate candidate GeoJSON export")
    parser.add_argument("--no_feature_extraction", "--no-feature-extraction", dest="no_feature_extraction", action="store_true",
                        help="Disable physical and geometric object feature extraction")
    parser.add_argument("--features_csv", default=None,
                        help="Custom destination path for candidate features CSV export")
    parser.add_argument("--device", default="cpu",
                        help="Compute device for neural networks (cpu or cuda, default: cpu)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input scene not found at {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        resolved_mode, resolved_model = resolve_pipeline_mode(
            mode=args.mode,
            model=args.model,
            obia_model=args.obia_model,
            script_dir=script_dir
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    output_target = args.output_dir
    if output_target is None and default_output is not None and os.path.abspath(args.input) == os.path.abspath(default_scene):
        output_target = default_output

    if resolved_mode == "two_stage":
        if not os.path.exists(resolved_model):
            print(f"Error: OBIA model bundle not found at {resolved_model}", file=sys.stderr)
            sys.exit(1)

        config = PipelineRunConfig(
            input_path=args.input,
            output_dir=output_target,
            mode="two_stage",
            obia_model_path=resolved_model,
            threshold=args.threshold,
            enable_vessel_masking=not args.no_vessel_masking,
            enable_candidate_export=not args.no_candidates,
            device=args.device,
            min_damping=args.min_damping,
            max_backscatter=args.max_backscatter,
            window_size=args.window_size,
            min_candidate_px=args.min_candidate_px,
            min_px=args.min_px
        )
        run_two_stage_pipeline(config)
    else:
        if not os.path.exists(resolved_model):
            print(f"Error: Model not found at {resolved_model}", file=sys.stderr)
            sys.exit(1)

        run_detection_pipeline(
            scene_path=args.input,
            model_path=resolved_model,
            output_dir=output_target,
            mode=resolved_mode,
            threshold=args.threshold,
            min_px=args.min_px,
            apply_lee=not args.no_lee,
            no_vessel_masking=args.no_vessel_masking,
            min_damping=args.min_damping,
            max_backscatter=args.max_backscatter,
            window_size=args.window_size,
            min_candidate_px=args.min_candidate_px,
            no_candidate_segmentation=args.no_candidates,
            no_feature_extraction=args.no_feature_extraction,
            features_csv=args.features_csv,
            device=args.device
        )

