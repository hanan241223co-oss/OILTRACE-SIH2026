#!/usr/bin/env python3
"""
OILTRACE Physical & Geometric Object Feature Extractor
------------------------------------------------------
Computes standardized 17-dimensional object feature vectors for candidate dark spot
polygons segmented from Sentinel-1 synthetic aperture radar scenes.

Feature Vector (17 features):
  1.  area_km2: Geodesic polygon area in square kilometers via WGS84 ellipsoid
  2.  perimeter_km: Geodesic perimeter in kilometers
  3.  elongation: Major axis / minor axis of minimum rotated rectangle (>= 1.0)
  4.  circularity: Isoperimetric quotient 4 * pi * area / perimeter^2 in (0.0, 1.0]
  5.  complexity_ratio: Perimeter to square root area ratio (P / sqrt(A))
  6.  fractal_dimension: Pixel space perimeter area complexity 2 * ln(P_px / 4) / ln(A_px) in [1.0, 2.0]
  7.  bbox_extent: Candidate pixel area / bounding box pixel area in (0.0, 1.0]
  8.  mean_vv_db: Mean VV backscatter in decibels inside candidate
  9.  mean_vh_db: Mean VH backscatter in decibels inside candidate
  10. min_vv_db: Peak dark core minimum VV backscatter in decibels
  11. vv_vh_ratio: Polarimetric decibel difference: mean_vv_db - mean_vh_db
  12. std_vv_db: Standard deviation of VV backscatter inside slick (texture)
  13. local_damping_db: Ambient ocean clutter mean minus slick mean VV in decibels
  14. boundary_gradient_mean: Mean Sobel gradient magnitude on valid 3 px perimeter ring
  15. boundary_gradient_max: 95th percentile Sobel gradient magnitude on perimeter ring
  16. distance_to_nearest_vessel_m: Distance in meters to closest vessel (10000.0 if no vessels)
  17. touches_suspect_vessel: 1.0 if candidate is within 35 px of suspect polluter, else 0.0
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, find_objects, sobel

try:
    from shapely.geometry import Polygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

try:
    import pyproj
    from pyproj import Geod
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

try:
    from pipeline.segmentation import CandidateDarkSpot, SegmentationResult
except ImportError:
    try:
        from .segmentation import CandidateDarkSpot, SegmentationResult
    except ImportError:
        from segmentation import CandidateDarkSpot, SegmentationResult


# ==============================================================================
# Canonical 17-Dimensional Feature Vector Specification
# ==============================================================================

CANONICAL_FEATURE_NAMES: list[str] = [
    # 1. Geometric shape morphology (7 features)
    "area_km2",
    "perimeter_km",
    "elongation",
    "circularity",
    "complexity_ratio",
    "fractal_dimension",
    "bbox_extent",
    # 2. Physical radar backscatter (6 features)
    "mean_vv_db",
    "mean_vh_db",
    "min_vv_db",
    "vv_vh_ratio",
    "std_vv_db",
    "local_damping_db",
    # 3. Interfacial boundary gradient (2 features)
    "boundary_gradient_mean",
    "boundary_gradient_max",
    # 4. Spatial contextual proximity (2 features)
    "distance_to_nearest_vessel_m",
    "touches_suspect_vessel",
]


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class CandidateObjectFeatures:
    """Structured container for candidate polygon physical and geometric features."""
    candidate_id: str                          # Identifier matching CandidateDarkSpot (e.g. spot_001)

    # 1. Geometric shape morphology (7 features)
    area_km2: float                            # Geodesic area in square kilometers
    perimeter_km: float                        # Geodesic perimeter in kilometers
    elongation: float                          # Aspect ratio of minimum rotated box (major / minor >= 1.0)
    circularity: float                         # 4 * pi * area / perimeter^2 in (0.0, 1.0]
    complexity_ratio: float                    # perimeter / sqrt(area)
    fractal_dimension: float                   # 2 * ln(P_px / 4) / ln(A_px) clamped to [1.0, 2.0]
    bbox_extent: float                         # Candidate pixel area / bounding box pixel area

    # 2. Physical radar backscatter (6 features)
    mean_vv_db: float                          # Mean VV backscatter in decibels
    mean_vh_db: float                          # Mean VH backscatter in decibels
    min_vv_db: float                           # Peak dark core minimum VV backscatter in decibels
    vv_vh_ratio: float                         # Polarimetric difference: mean_vv_db - mean_vh_db
    std_vv_db: float                           # Standard deviation of VV backscatter inside slick
    local_damping_db: float                    # Relative damping contrast against local ocean clutter

    # 3. Interfacial boundary gradient (2 features)
    boundary_gradient_mean: float              # Mean Sobel gradient magnitude on valid 3 px perimeter ring
    boundary_gradient_max: float               # 95th percentile Sobel gradient magnitude on boundary

    # 4. Spatial contextual proximity (2 features)
    distance_to_nearest_vessel_m: float        # Distance to closest vessel in meters (10000.0 if none)
    touches_suspect_vessel: float              # 1.0 if within 35 px of suspect polluter, else 0.0

    # Structured dictionary containing all 17 features in canonical order
    feature_vector: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Populate feature_vector dictionary if not explicitly supplied."""
        if not self.feature_vector:
            self.feature_vector = {k: float(getattr(self, k)) for k in CANONICAL_FEATURE_NAMES}

    def to_array(self) -> np.ndarray:
        """Returns 1D float32 NumPy array containing the 17 features in canonical order."""
        return np.array([self.feature_vector[k] for k in CANONICAL_FEATURE_NAMES], dtype=np.float32)

    def to_numpy_vector(self) -> np.ndarray:
        """Convenience alias returning 1D float32 NumPy array in canonical order."""
        return self.to_array()

    def to_feature_dict(self) -> dict[str, float]:
        """Returns ordered dictionary mapping all 17 canonical feature names to values."""
        return dict(self.feature_vector)


@dataclass
class FeatureExtractionResult:
    """Structured container encapsulating scene object feature extraction outputs."""
    scene_id: str                              # Input scene identifier
    candidate_count: int                       # Total candidates evaluated
    feature_names: list[str]                   # Ordered list of 17 canonical feature keys
    feature_matrix: np.ndarray                 # 2D float32 array of shape (candidate_count, 17)
    candidate_features: list[CandidateObjectFeatures] # List of extracted feature records


# ==============================================================================
# Geometric Morphology Calculations (AC-1)
# ==============================================================================

def compute_polygon_geometric_features(contour_px: list[tuple[int, int]],
                                       pixel_count: int,
                                       bbox_px: tuple[int, int, int, int],
                                       area_km2: float,
                                       perimeter_km: float) -> dict[str, float]:
    """
    Computes 7 geometric shape morphology descriptors:
    - area_km2, perimeter_km
    - elongation: aspect ratio of minimum bounding rotated box (major / minor >= 1.0)
    - circularity: isoperimetric quotient 4 * pi * area / perimeter^2
    - complexity_ratio: perimeter / sqrt(area)
    - fractal_dimension: 2 * ln(P_px / 4) / ln(A_px) in [1.0, 2.0]
    - bbox_extent: pixel_count / bbox_pixel_area
    """
    # Safe defaults
    elongation = 1.0
    circularity = 0.0
    complexity_ratio = 0.0
    fractal_dimension = 1.0
    bbox_extent = 1.0

    # 1. Elongation via minimum rotated bounding rectangle in pixel space
    if HAS_SHAPELY and len(contour_px) >= 3:
        try:
            poly = Polygon(contour_px)
            if poly.is_valid and not poly.is_empty:
                min_rect = poly.minimum_rotated_rectangle
                rect_coords = list(min_rect.exterior.coords)
                if len(rect_coords) >= 4:
                    # Calculate side lengths
                    s1 = math.hypot(rect_coords[1][0] - rect_coords[0][0], rect_coords[1][1] - rect_coords[0][1])
                    s2 = math.hypot(rect_coords[2][0] - rect_coords[1][0], rect_coords[2][1] - rect_coords[1][1])
                    major = max(s1, s2)
                    minor = min(s1, s2)
                    if minor > 1e-3:
                        elongation = float(np.clip(major / minor, 1.0, 50.0))
                    else:
                        elongation = 50.0 if major > 0 else 1.0
        except Exception:
            elongation = 1.0

    # 2. Circularity compactness: 4 * pi * A / P^2
    if perimeter_km > 1e-6 and area_km2 > 1e-9:
        # Convert both to square meters and meters for consistent ratio
        area_m2 = area_km2 * 1e6
        perim_m = perimeter_km * 1e3
        circ_val = (4.0 * math.pi * area_m2) / (perim_m * perim_m)
        circularity = float(np.clip(circ_val, 0.0, 1.0))

    # 3. Complexity ratio: P / sqrt(A)
    if area_km2 > 1e-9 and perimeter_km > 1e-6:
        area_m2 = area_km2 * 1e6
        perim_m = perimeter_km * 1e3
        complexity_ratio = float(perim_m / math.sqrt(area_m2))

    # 4. Pixel space perimeter area fractal dimension: 2 * ln(P_px / 4) / ln(A_px)
    if pixel_count >= 2:
        # Approximate pixel perimeter from contour length or polygon boundary
        p_px = float(len(contour_px)) if len(contour_px) > 0 else math.sqrt(pixel_count) * 4.0
        p_quarter = max(p_px / 4.0, 1.001)
        a_px = max(float(pixel_count), 2.0)
        frac_val = 2.0 * math.log(p_quarter) / math.log(a_px)
        fractal_dimension = float(np.clip(frac_val, 1.0, 2.0))

    # 5. Bounding box extent: candidate pixel area / bounding box area
    col_min, row_min, col_max, row_max = bbox_px
    bbox_w = max(col_max - col_min, 1)
    bbox_h = max(row_max - row_min, 1)
    bbox_area = bbox_w * bbox_h
    bbox_extent = float(np.clip(pixel_count / float(bbox_area), 0.0, 1.0))

    return {
        "area_km2": float(area_km2),
        "perimeter_km": float(perimeter_km),
        "elongation": elongation,
        "circularity": circularity,
        "complexity_ratio": complexity_ratio,
        "fractal_dimension": fractal_dimension,
        "bbox_extent": bbox_extent,
    }


# ==============================================================================
# Boundary Gradient Sharpness Calculation (AC-3)
# ==============================================================================

def compute_boundary_gradient_sharpness(spot_slice_mask: np.ndarray,
                                        sobel_mag_slice: np.ndarray,
                                        valid_slice: np.ndarray) -> tuple[float, float]:
    """
    Computes mean and 95th percentile Sobel gradient magnitude along a 3-pixel outer
    dilated perimeter ring, masked by ocean validity to prevent artificial nodata edge steps.
    """
    if spot_slice_mask.size == 0 or not np.any(spot_slice_mask):
        return 0.0, 0.0

    # 3-pixel outer dilation
    struct_3x3 = np.ones((3, 3), dtype=bool)
    dilated_mask = binary_dilation(spot_slice_mask, structure=struct_3x3, iterations=3)

    # Perimeter ring: pixels in dilated mask that are NOT inside the slick and are valid ocean
    boundary_ring = dilated_mask & (~spot_slice_mask) & valid_slice

    if not np.any(boundary_ring):
        # Fallback: if exterior ring is masked by invalid borders, check slick outer border itself
        eroded_mask = binary_dilation(~spot_slice_mask, structure=struct_3x3, iterations=1)
        fallback_ring = spot_slice_mask & eroded_mask & valid_slice
        if np.any(fallback_ring):
            vals = sobel_mag_slice[fallback_ring]
            return float(np.mean(vals)), float(np.percentile(vals, 95.0))
        return 0.0, 0.0

    gradient_vals = sobel_mag_slice[boundary_ring]
    mean_gradient = float(np.mean(gradient_vals))
    max_gradient = float(np.percentile(gradient_vals, 95.0))

    return mean_gradient, max_gradient


# ==============================================================================
# Full Candidate Feature Extraction Pipeline (AC-1 through AC-5)
# ==============================================================================

def extract_candidate_features(candidates: list[CandidateDarkSpot],
                               candidate_mask: np.ndarray,
                               sar_data: np.ndarray,
                               damping_map: np.ndarray,
                               metadata: dict,
                               vessels: Optional[list] = None) -> FeatureExtractionResult:
    """
    Extracts standardized 17-dimensional object feature vectors for all candidate dark spots:
    1. Precomputes full scene Sobel gradient magnitude on VV channel for fast boundary sampling.
    2. Uses slice objects from candidate_mask for fast localized array indexing.
    3. Computes geometric shape morphology (elongation, circularity, complexity, fractal dimension, extent).
    4. Computes radar backscatter statistics (mean VV/VH, peak min VV, polarimetric ratio, texture std).
    5. Computes boundary transition gradient sharpness (mean and 95th percentile).
    6. Incorporates vessel proximity context and suspect discharge linkage.
    7. Assembles typed CandidateObjectFeatures and (N, 17) NumPy feature matrix.
    """
    scene_id = str(metadata.get("path", metadata.get("scene_id", "scene")))
    n_candidates = len(candidates)

    # Empty candidate scene handling
    if n_candidates == 0 or candidate_mask.size == 0 or not np.any(candidate_mask):
        empty_matrix = np.empty((0, 17), dtype=np.float32)
        return FeatureExtractionResult(
            scene_id=scene_id,
            candidate_count=0,
            feature_names=CANONICAL_FEATURE_NAMES,
            feature_matrix=empty_matrix,
            candidate_features=[]
        )

    # Ensure sar_data is H x W x C
    if sar_data.ndim == 3 and sar_data.shape[0] == 2:
        sar_data = np.transpose(sar_data, (1, 2, 0))

    H, W = sar_data.shape[:2]
    vv_db = np.clip(sar_data[:, :, 0].astype(np.float32), -50.0, 5.0)
    vh_db = np.clip(sar_data[:, :, 1].astype(np.float32), -50.0, 5.0) if sar_data.shape[-1] > 1 else vv_db
    valid_mask = ((vv_db != 0.0) | (vh_db != 0.0)) & ~np.isnan(vv_db)

    # 1. Precompute full-scene 3x3 Sobel gradient magnitude on VV channel
    sobel_y = sobel(vv_db, axis=0)
    sobel_x = sobel(vv_db, axis=1)
    sobel_mag = np.hypot(sobel_x, sobel_y).astype(np.float32)

    # Zero out artificial boundary steps outside valid ocean
    sobel_mag[~valid_mask] = 0.0

    # 2. Slice objects for fast component extraction
    slices = find_objects(candidate_mask)

    extracted_records: list[CandidateObjectFeatures] = []
    matrix_rows: list[np.ndarray] = []

    for cand in candidates:
        lbl = cand.label_id

        # Local raster slice for localized computations
        slice_obj = slices[lbl - 1] if (0 < lbl <= len(slices)) and (slices[lbl - 1] is not None) else None

        if slice_obj is not None:
            # Expand slice slightly (4 pixels) to accommodate 3-pixel dilated boundary ring
            row_s, col_s = slice_obj
            r_min = max(row_s.start - 4, 0)
            r_max = min(row_s.stop + 4, H)
            c_min = max(col_s.start - 4, 0)
            c_max = min(col_s.stop + 4, W)
            exp_slice = (slice(r_min, r_max), slice(c_min, c_max))

            local_labeled = candidate_mask[exp_slice]
            spot_slice_mask = (local_labeled == lbl)
            sobel_slice = sobel_mag[exp_slice]
            valid_slice = valid_mask[exp_slice]
            vv_slice = vv_db[exp_slice]
        else:
            spot_slice_mask = np.zeros((1, 1), dtype=bool)
            sobel_slice = np.zeros((1, 1), dtype=np.float32)
            valid_slice = np.zeros((1, 1), dtype=bool)
            vv_slice = np.zeros((1, 1), dtype=np.float32)

        # 3. Geometric Shape Morphology (AC-1)
        # Approximate geodesic perimeter from area and contour if not provided
        perim_km = getattr(cand, "perimeter_km", None)
        if perim_km is None:
            # Estimate perimeter from pixel count and pixel spacing (10m)
            perim_km = float(len(cand.contour_px) * 0.010) if len(cand.contour_px) > 0 else float(math.sqrt(cand.area_km2) * 4.0)

        geom_metrics = compute_polygon_geometric_features(
            contour_px=cand.contour_px,
            pixel_count=cand.pixel_count,
            bbox_px=cand.bbox_px,
            area_km2=cand.area_km2,
            perimeter_km=perim_km
        )

        # 4. Radar Backscatter & Texture Statistics (AC-2)
        mean_vv = float(cand.mean_vv_db)
        mean_vh = float(cand.mean_vh_db)
        min_vv = float(cand.min_vv_db)
        local_damping = float(cand.local_damping_db)
        vv_vh_ratio = float(mean_vv - mean_vh)

        # Texture standard deviation inside candidate slick
        if np.any(spot_slice_mask) and cand.pixel_count >= 2:
            std_vv = float(np.std(vv_slice[spot_slice_mask]))
        else:
            std_vv = 0.0

        # 5. Interfacial Boundary Gradient Sharpness (AC-3)
        bg_mean, bg_max = compute_boundary_gradient_sharpness(
            spot_slice_mask=spot_slice_mask,
            sobel_mag_slice=sobel_slice,
            valid_slice=valid_slice
        )

        # 6. Spatial Contextual Vessel Proximity (AC-4)
        dist_vessel = float(cand.distance_to_nearest_vessel_m) if cand.distance_to_nearest_vessel_m is not None else None
        touches_vessel = 1.0 if cand.touches_suspect_vessel else 0.0

        if (dist_vessel is None or dist_vessel >= 10000.0) and vessels:
            px_size = max(metadata.get('pixel_size_x', 10.0), 1.0)
            cx, cy = cand.centroid_px
            min_dist_m = 10000.0
            for v in vessels:
                vx = getattr(v, 'pixel_x', getattr(v, 'centroid_col', getattr(v, 'col', 0.0)))
                vy = getattr(v, 'pixel_y', getattr(v, 'centroid_row', getattr(v, 'row', 0.0)))
                d_px = math.hypot(cx - vx, cy - vy)
                d_m = d_px * px_size
                if d_m < min_dist_m:
                    min_dist_m = d_m
                if getattr(v, 'is_suspect_source', False) and d_px <= 35.0:
                    touches_vessel = 1.0
            dist_vessel = min_dist_m

        if dist_vessel is None:
            dist_vessel = 10000.0

        # 7. Assemble 17-Dimensional Feature Dictionary
        feature_dict: dict[str, float] = {
            "area_km2": geom_metrics["area_km2"],
            "perimeter_km": geom_metrics["perimeter_km"],
            "elongation": geom_metrics["elongation"],
            "circularity": geom_metrics["circularity"],
            "complexity_ratio": geom_metrics["complexity_ratio"],
            "fractal_dimension": geom_metrics["fractal_dimension"],
            "bbox_extent": geom_metrics["bbox_extent"],
            "mean_vv_db": mean_vv,
            "mean_vh_db": mean_vh,
            "min_vv_db": min_vv,
            "vv_vh_ratio": vv_vh_ratio,
            "std_vv_db": std_vv,
            "local_damping_db": local_damping,
            "boundary_gradient_mean": bg_mean,
            "boundary_gradient_max": bg_max,
            "distance_to_nearest_vessel_m": dist_vessel,
            "touches_suspect_vessel": touches_vessel,
        }

        # Check for NaN / Inf safety
        for k, v in feature_dict.items():
            if math.isnan(v) or math.isinf(v):
                feature_dict[k] = 0.0

        feat_obj = CandidateObjectFeatures(
            candidate_id=cand.candidate_id,
            area_km2=feature_dict["area_km2"],
            perimeter_km=feature_dict["perimeter_km"],
            elongation=feature_dict["elongation"],
            circularity=feature_dict["circularity"],
            complexity_ratio=feature_dict["complexity_ratio"],
            fractal_dimension=feature_dict["fractal_dimension"],
            bbox_extent=feature_dict["bbox_extent"],
            mean_vv_db=feature_dict["mean_vv_db"],
            mean_vh_db=feature_dict["mean_vh_db"],
            min_vv_db=feature_dict["min_vv_db"],
            vv_vh_ratio=feature_dict["vv_vh_ratio"],
            std_vv_db=feature_dict["std_vv_db"],
            local_damping_db=feature_dict["local_damping_db"],
            boundary_gradient_mean=feature_dict["boundary_gradient_mean"],
            boundary_gradient_max=feature_dict["boundary_gradient_max"],
            distance_to_nearest_vessel_m=feature_dict["distance_to_nearest_vessel_m"],
            touches_suspect_vessel=feature_dict["touches_suspect_vessel"],
            feature_vector=feature_dict
        )

        # Attach to candidate entity
        cand.features = feat_obj

        extracted_records.append(feat_obj)
        matrix_rows.append(feat_obj.to_array())

    feature_matrix = np.vstack(matrix_rows).astype(np.float32) if matrix_rows else np.empty((0, 17), dtype=np.float32)

    return FeatureExtractionResult(
        scene_id=scene_id,
        candidate_count=len(extracted_records),
        feature_names=CANONICAL_FEATURE_NAMES,
        feature_matrix=feature_matrix,
        candidate_features=extracted_records
    )


# ==============================================================================
# Tabular Export to Pandas DataFrame & CSV (AC-5)
# ==============================================================================

def export_features_tabular(result: FeatureExtractionResult,
                            output_csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Exports extracted object features into a pandas DataFrame ready for scikit-learn models.
    Optionally saves to CSV file if output_csv_path is specified.
    """
    if result.candidate_count == 0 or not result.candidate_features:
        df = pd.DataFrame(columns=["candidate_id"] + CANONICAL_FEATURE_NAMES)
        if output_csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
            df.to_csv(output_csv_path, index=False)
        return df

    rows = []
    for feat in result.candidate_features:
        row = {"candidate_id": feat.candidate_id}
        row.update(feat.feature_vector)
        rows.append(row)

    df = pd.DataFrame(rows)

    if output_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        df.to_csv(output_csv_path, index=False)

    return df
