#!/usr/bin/env python3
"""
OILTRACE Adaptive Dark Spot Candidate Segmentation Engine
---------------------------------------------------------
Extracts connected candidate oil spill polygons from synthetic aperture radar imagery.
Calculates moving window ambient ocean clutter to determine local radar damping contrast,
labels candidate components using 8-connectivity, integrates vessel shadow suppression
to protect genuine discharging slicks, and exports georeferenced candidate vector polygons.
"""

import os
import sys
import math
import json
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
from scipy import ndimage
from scipy.ndimage import (
    uniform_filter,
    label,
    find_objects,
    binary_closing,
    binary_dilation,
    binary_erosion,
)

# Geospatial libraries
try:
    import rasterio
    import rasterio.features
    from rasterio.transform import xy
    from rasterio.warp import transform_geom, transform
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import shapely.geometry
    from shapely.geometry import Polygon, MultiPolygon, shape, mapping, box
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

try:
    import pyproj
    from pyproj import Geod
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

# Autonomous vessel reflector and shadow corridor masking
try:
    from pipeline.vessel_masking import mask_vessel_shadows, VesselTarget
except ImportError:
    try:
        from .vessel_masking import mask_vessel_shadows, VesselTarget
    except ImportError:
        try:
            from vessel_masking import mask_vessel_shadows, VesselTarget
        except ImportError:
            mask_vessel_shadows = None
            VesselTarget = None


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class CandidateDarkSpot:
    """Represents a discrete candidate oil spill dark spot polygon."""
    candidate_id: str                      # Unique identifier (e.g. spot_001)
    label_id: int                          # Consecutive integer label in segmentation mask (1 to N)
    pixel_count: int                       # Total pixel count
    area_km2: float                        # Geodesic area in square kilometers calculated via pyproj.Geod
    bbox_px: tuple[int, int, int, int]     # Half open raster bounding box (col_min, row_min, col_max, row_max)
    bbox_wgs84: tuple[float, float, float, float] # Geographic bounding box (lon_min, lat_min, lon_max, lat_max)
    geometry_wgs84: dict                   # GeoJSON polygon coordinate geometry (EPSG 4326)
    contour_px: list[tuple[int, int]]      # Pixel boundary coordinates as (col, row) tuples
    centroid_px: tuple[float, float]       # Centroid column and row coordinates
    centroid_wgs84: tuple[float, float]    # Centroid longitude and latitude
    mean_vv_db: float                      # Average VV backscatter in decibels
    min_vv_db: float                       # Peak dark core VV backscatter in decibels
    mean_vh_db: float                      # Average VH backscatter in decibels
    local_damping_db: float                # Mean damping ratio of candidate pixels relative to ambient ocean
    nearest_vessel_id: Optional[str] = None # Identifier of nearest vessel target if detected
    distance_to_nearest_vessel_m: Optional[float] = None # Distance in meters to closest detected vessel
    touches_suspect_vessel: bool = False   # True if candidate boundary is within 35 px of a suspect discharging vessel
    is_confirmed_slick: Optional[bool] = None # Set by Stage 2 classifier
    slick_probability: Optional[float] = None # Set by Stage 2 classifier
    features: Optional[Any] = None         # CandidateObjectFeatures attached by feature extractor


@dataclass
class SegmentationResult:
    """Structured container encapsulating scene segmentation outputs."""
    scene_id: str                          # Name of input scene
    candidate_count: int                   # Total candidate dark spots extracted
    total_candidate_area_km2: float        # Sum of candidate geodesic areas
    candidate_mask: np.ndarray             # 2D integer array of labeled candidate spots (0 = ocean, 1..N = spot)
    damping_ratio_map: np.ndarray          # 2D float32 array of relative damping contrast values (0.0 for invalid)
    candidates: list[CandidateDarkSpot]     # Extracted candidate objects sorted by area descending


# ==============================================================================
# Background Ocean & Damping Ratio Calculation
# ==============================================================================

def compute_adaptive_ocean_background(vv_db: np.ndarray,
                                      valid_mask: np.ndarray,
                                      window_size: int = 151,
                                      min_samples: int = 100) -> np.ndarray:
    """
    Computes local ambient ocean clutter using a moving window box filter normalized
    exclusively over valid ocean pixels. Falls back to global ocean median if a local
    window contains fewer than min_samples valid pixels (e.g. coastal boundaries).
    """
    if not np.any(valid_mask):
        return np.full_like(vv_db, -15.0, dtype=np.float32)

    valid_float = valid_mask.astype(np.float32)
    masked_vv = np.where(valid_mask, vv_db, 0.0).astype(np.float32)

    # Scipy uniform_filter computes box mean: mean = sum / (window_size * window_size)
    win_area = float(window_size * window_size)
    mean_val = uniform_filter(masked_vv, size=window_size, mode='reflect')
    mean_cnt = uniform_filter(valid_float, size=window_size, mode='reflect')

    sum_arr = mean_val * win_area
    cnt_arr = mean_cnt * win_area

    # Calculate global ocean median for safe coastal fallback
    valid_vv = vv_db[valid_mask]
    global_median = float(np.median(valid_vv)) if valid_vv.size > 0 else -15.0

    # Avoid division by zero: divide where valid ocean pixel count exceeds min_samples
    has_enough_samples = cnt_arr >= float(min_samples)

    # Vectorized assignment
    local_mean = np.where(
        has_enough_samples,
        sum_arr / np.maximum(cnt_arr, 1.0),
        global_median
    )

    return local_mean.astype(np.float32)


def compute_damping_ratio_map(vv_db: np.ndarray,
                              local_ocean_mean: np.ndarray,
                              valid_mask: np.ndarray) -> np.ndarray:
    """
    Computes pixel level relative damping contrast in decibels:
    DR = local_ocean_mean - vv_db.
    Invalid or non ocean pixels are assigned 0.0 dB.
    """
    damping = local_ocean_mean - vv_db
    damping_map = np.where(valid_mask, damping, 0.0).astype(np.float32)
    return damping_map


# ==============================================================================
# Candidate Dark Spot Extraction & Vessel Linkage
# ==============================================================================

def segment_candidate_dark_spots(sar_data: np.ndarray,
                                 metadata: dict,
                                 min_damping_db: float = 2.2,
                                 max_backscatter_db: float = -20.0,
                                 min_area_px: int = 30,
                                 window_size: int = 151,
                                 vessels: Optional[list] = None,
                                 vessel_mask: Optional[np.ndarray] = None) -> SegmentationResult:
    """
    Main extraction pipeline:
    1. Calibrates VV and VH channels and creates ocean validity mask.
    2. Computes moving window ocean background and relative damping ratio map.
    3. Generates preliminary candidate mask using damping contrast and decibel ceiling.
    4. Applies 3x3 morphological closing and prunes speckle noise (< min_area_px).
    5. Passes coherent candidate mask to vessel masking to suppress radar shadow
       corridors while preserving genuine discharging oil slicks.
    6. Labels surviving components with 8-connectivity and renumbers consecutively (1..N).
    7. Fast vectorized polygonization via rasterio shapes and geodesic area calculation via pyproj.
    8. Records proximity to detected vessels and suspect discharging ships.
    """
    if sar_data.ndim == 3 and sar_data.shape[0] == 2:
        sar_data = np.transpose(sar_data, (1, 2, 0))

    H, W = sar_data.shape[:2]
    vv_raw = sar_data[:, :, 0].astype(np.float32)
    vh_raw = sar_data[:, :, 1].astype(np.float32) if sar_data.ndim == 3 and sar_data.shape[2] > 1 else vv_raw

    valid_mask = ((vv_raw != 0.0) | (vh_raw != 0.0)) & ~np.isnan(vv_raw)
    vv_db = np.clip(vv_raw, -50.0, 5.0)
    vh_db = np.clip(vh_raw, -50.0, 5.0)

    # 1. Moving window background and damping ratio calculation
    local_ocean_mean = compute_adaptive_ocean_background(
        vv_db, valid_mask, window_size=window_size, min_samples=100
    )
    damping_map = compute_damping_ratio_map(vv_db, local_ocean_mean, valid_mask)

    # 2. Preliminary thresholding
    preliminary_mask = valid_mask & (damping_map >= min_damping_db) & (vv_db <= max_backscatter_db)

    # 3. Morphological closing to bridge minor internal voids
    closing_structure = np.ones((3, 3), dtype=bool)
    closed_mask = binary_closing(preliminary_mask, structure=closing_structure)

    # 4. Prune single-pixel speckle noise before vessel shadow evaluation
    connectivity_8 = np.ones((3, 3), dtype=int)
    labeled_pre, num_pre = label(closed_mask, structure=connectivity_8)

    if num_pre > 0:
        counts_pre = np.bincount(labeled_pre.ravel())
        valid_comp_ids = np.where(counts_pre >= min_area_px)[0]
        valid_comp_ids = valid_comp_ids[valid_comp_ids > 0]
        coarse_candidate_mask = np.isin(labeled_pre, valid_comp_ids)
    else:
        coarse_candidate_mask = np.zeros((H, W), dtype=bool)

    # 5. Vessel shadow masking and slick preservation integration
    detected_vessels = vessels if vessels is not None else []
    if mask_vessel_shadows is not None and np.any(coarse_candidate_mask):
        if vessel_mask is not None:
            cleaned_mask = coarse_candidate_mask & (~vessel_mask.astype(bool))
        else:
            # Evaluates shadow corridors on candidate objects, preserving genuine slicks
            cleaned_mask, vessel_targets, _ = mask_vessel_shadows(
                sar_data, coarse_candidate_mask, metadata
            )
            if vessels is None:
                detected_vessels = vessel_targets
    else:
        cleaned_mask = coarse_candidate_mask

    # 6. Final connected component labeling using 8-connectivity
    labeled_features, num_features = label(cleaned_mask, structure=connectivity_8)

    # Coordinate transforms and Geod geodesic area calculator
    gt = metadata.get('transform')
    crs = metadata.get('crs')
    px_size_x = float(metadata.get('pixel_size_x', 10.0))
    px_size_y = float(metadata.get('pixel_size_y', 10.0))
    planar_px_area_km2 = (px_size_x * px_size_y) / 1e6

    geod = None
    if HAS_PYPROJ:
        try:
            geod = Geod(ellps='WGS84')
        except Exception:
            geod = None

    raw_candidates = []
    if num_features > 0:
        counts = np.bincount(labeled_features.ravel())
        slices = find_objects(labeled_features)

        for lbl_idx in range(1, num_features + 1):
            px_count = int(counts[lbl_idx])
            if px_count < min_area_px:
                continue

            slc = slices[lbl_idx - 1]
            if slc is None:
                continue

            r_slice, c_slice = slc
            comp_crop = (labeled_features[r_slice, c_slice] == lbl_idx)
            sub_rows, sub_cols = np.where(comp_crop)

            rows = sub_rows + r_slice.start
            cols = sub_cols + c_slice.start

            r_min, r_max = int(rows.min()), int(rows.max() + 1)
            c_min, c_max = int(cols.min()), int(cols.max() + 1)
            bbox_px = (c_min, r_min, c_max, r_max)

            cy_px = float(rows.mean())
            cx_px = float(cols.mean())
            centroid_px = (cx_px, cy_px)

            # Radiometric statistics
            comp_vv = vv_db[r_slice, c_slice][comp_crop]
            comp_vh = vh_db[r_slice, c_slice][comp_crop]
            comp_damping = damping_map[r_slice, c_slice][comp_crop]

            mean_vv_db = float(np.mean(comp_vv))
            min_vv_db = float(np.min(comp_vv))
            mean_vh_db = float(np.mean(comp_vh))
            local_damping_db = float(np.mean(comp_damping))

            # Outer boundary contour coordinates as (col, row)
            border_crop = comp_crop & (~binary_erosion(comp_crop, structure=closing_structure))
            b_rows_sub, b_cols_sub = np.where(border_crop)
            b_rows = b_rows_sub + r_slice.start
            b_cols = b_cols_sub + c_slice.start
            contour_px = list(zip(b_cols.tolist(), b_rows.tolist()))

            # Vessel proximity and suspect ship linkage
            nearest_vessel_id = None
            min_dist_m = None
            touches_suspect = False

            if detected_vessels:
                min_dist_px = float('inf')
                for v in detected_vessels:
                    dist_px = math.hypot(v.pixel_x - cx_px, v.pixel_y - cy_px)
                    if dist_px < min_dist_px:
                        min_dist_px = dist_px
                        nearest_vessel_id = v.target_id

                    # Check proximity to suspect discharging vessel within 35 pixels
                    if getattr(v, 'is_suspect_source', False) and dist_px <= 35.0:
                        touches_suspect = True

                # Calculate physical distance in meters
                if min_dist_px != float('inf'):
                    min_dist_m = float(min_dist_px * math.sqrt(px_size_x * px_size_y))

            raw_candidates.append({
                'pixel_count': px_count,
                'planar_area_km2': float(px_count * planar_px_area_km2),
                'bbox_px': bbox_px,
                'contour_px': contour_px,
                'centroid_px': centroid_px,
                'mean_vv_db': mean_vv_db,
                'min_vv_db': min_vv_db,
                'mean_vh_db': mean_vh_db,
                'local_damping_db': local_damping_db,
                'nearest_vessel_id': nearest_vessel_id,
                'distance_to_nearest_vessel_m': min_dist_m,
                'touches_suspect_vessel': touches_suspect,
                'orig_label': lbl_idx
            })

    # Sort surviving candidates descending by planar area first
    raw_candidates.sort(key=lambda c: c['planar_area_km2'], reverse=True)

    # Re-index surviving candidate components consecutively from 1 to N
    candidate_mask = np.zeros((H, W), dtype=np.int32)
    for new_id, item in enumerate(raw_candidates, start=1):
        orig_lbl = item['orig_label']
        candidate_mask[labeled_features == orig_lbl] = new_id

    # Fast vectorized polygonization via rasterio shapes across the labeled mask
    poly_wgs84_map = {}
    if HAS_RASTERIO and HAS_SHAPELY and gt and np.any(candidate_mask > 0):
        try:
            shapes_gen = rasterio.features.shapes(
                candidate_mask,
                mask=(candidate_mask > 0),
                transform=gt
            )
            for geom_dict, label_val in shapes_gen:
                lbl_int = int(label_val)
                if lbl_int <= 0:
                    continue

                raw_poly = shape(geom_dict)
                # Reproject to WGS84 EPSG:4326 if necessary
                if crs and str(crs).upper() not in ['EPSG:4326', 'WGS84']:
                    try:
                        transformed_dict = transform_geom(crs, 'EPSG:4326', geom_dict)
                        wgs_poly = shape(transformed_dict)
                    except Exception:
                        wgs_poly = raw_poly
                else:
                    wgs_poly = raw_poly

                # If component was split into multi-part shapes, union them
                if lbl_int in poly_wgs84_map:
                    poly_wgs84_map[lbl_int] = poly_wgs84_map[lbl_int].union(wgs_poly)
                else:
                    poly_wgs84_map[lbl_int] = wgs_poly
        except Exception:
            poly_wgs84_map = {}

    final_candidates = []
    for new_id, item in enumerate(raw_candidates, start=1):
        item.pop('orig_label')
        planar_area = item.pop('planar_area_km2')

        wgs_poly = poly_wgs84_map.get(new_id)
        area_km2 = planar_area
        centroid_wgs84 = (0.0, 0.0)
        bbox_wgs84 = (0.0, 0.0, 0.0, 0.0)
        geometry_wgs84 = {}

        if wgs_poly is not None:
            geometry_wgs84 = mapping(wgs_poly)
            b_minx, b_miny, b_maxx, b_maxy = wgs_poly.bounds
            bbox_wgs84 = (float(b_minx), float(b_miny), float(b_maxx), float(b_maxy))
            centroid_wgs84 = (float(wgs_poly.centroid.x), float(wgs_poly.centroid.y))

            # Calculate geodesic area via pyproj Geod
            if geod is not None:
                try:
                    geo_area_m2, _ = geod.geometry_area_perimeter(wgs_poly)
                    area_km2 = float(abs(geo_area_m2) / 1e6)
                except Exception:
                    area_km2 = planar_area

        # Fallback centroid coordinates if geometry transform was unavailable
        if centroid_wgs84 == (0.0, 0.0) and gt and HAS_RASTERIO:
            try:
                cx_px, cy_px = item['centroid_px']
                map_x, map_y = rasterio.transform.xy(gt, cy_px, cx_px, offset='center')
                if crs:
                    lon, lat = transform(crs, 'EPSG:4326', [map_x], [map_y])
                    centroid_wgs84 = (float(lon[0]), float(lat[0]))
                else:
                    centroid_wgs84 = (float(map_x), float(map_y))
            except Exception:
                pass

        cand_obj = CandidateDarkSpot(
            candidate_id=f"spot_{new_id:03d}",
            label_id=new_id,
            area_km2=area_km2,
            bbox_wgs84=bbox_wgs84,
            geometry_wgs84=geometry_wgs84,
            centroid_wgs84=centroid_wgs84,
            **item
        )
        final_candidates.append(cand_obj)

    # Sort final candidates descending by geodesic area
    final_candidates.sort(key=lambda c: c.area_km2, reverse=True)
    # Re-assign sequential IDs after geodesic sorting
    for rank, c in enumerate(final_candidates, start=1):
        c.candidate_id = f"spot_{rank:03d}"

    scene_id = os.path.splitext(os.path.basename(metadata.get('path', 'unknown_scene')))[0]
    total_area_km2 = float(sum(c.area_km2 for c in final_candidates))

    return SegmentationResult(
        scene_id=scene_id,
        candidate_count=len(final_candidates),
        total_candidate_area_km2=round(total_area_km2, 4),
        candidate_mask=candidate_mask,
        damping_ratio_map=damping_map,
        candidates=final_candidates
    )


# ==============================================================================
# GeoJSON Export Functionality
# ==============================================================================

def export_candidate_geojson(candidates: list[CandidateDarkSpot],
                             metadata: dict,
                             output_path: str) -> None:
    """
    Exports candidate dark spot polygons to a standard GeoJSON FeatureCollection.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    features = []

    for c in candidates:
        geom = c.geometry_wgs84
        if not geom or 'coordinates' not in geom:
            b = c.bbox_wgs84
            if b != (0.0, 0.0, 0.0, 0.0):
                poly_coords = [
                    [b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]
                ]
                geom = {"type": "Polygon", "coordinates": [poly_coords]}
            else:
                c_min, r_min, c_max, r_max = c.bbox_px
                geom = {
                    "type": "Polygon",
                    "coordinates": [[[c_min, r_min], [c_max, r_min], [c_max, r_max], [c_min, r_max], [c_min, r_min]]]
                }

        properties = {
            "candidate_id": c.candidate_id,
            "label_id": c.label_id,
            "pixel_count": c.pixel_count,
            "area_km2": round(c.area_km2, 4),
            "area_hectares": round(c.area_km2 * 100.0, 2),
            "mean_vv_db": round(c.mean_vv_db, 2),
            "min_vv_db": round(c.min_vv_db, 2),
            "mean_vh_db": round(c.mean_vh_db, 2),
            "local_damping_db": round(c.local_damping_db, 2),
            "nearest_vessel_id": c.nearest_vessel_id,
            "distance_to_nearest_vessel_m": round(c.distance_to_nearest_vessel_m, 1) if c.distance_to_nearest_vessel_m is not None else None,
            "touches_suspect_vessel": c.touches_suspect_vessel,
            "centroid_px": [round(c.centroid_px[0], 1), round(c.centroid_px[1], 1)],
            "centroid_wgs84": [round(c.centroid_wgs84[0], 5), round(c.centroid_wgs84[1], 5)],
            "bbox_px": list(c.bbox_px),
            "bbox_wgs84": [round(coord, 5) for coord in c.bbox_wgs84]
        }

        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": properties
        })

    geojson_obj = {
        "type": "FeatureCollection",
        "name": "candidate_dark_spots",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson_obj, f, indent=2)


# ==============================================================================
# Standalone CLI Entrypoint
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive Dark Spot Candidate Segmentation")
    parser.add_argument("--scene", "-i", required=True, help="Input SAR GeoTIFF scene path")
    parser.add_argument("--output", "-o", default=None, help="Output directory for GeoJSON deliverable")
    parser.add_argument("--min_damping", type=float, default=2.2, help="Minimum damping contrast threshold (dB)")
    parser.add_argument("--max_backscatter", type=float, default=-20.0, help="Maximum VV backscatter ceiling (dB)")
    parser.add_argument("--min_area_px", type=int, default=30, help="Minimum candidate cluster size in pixels")
    parser.add_argument("--window_size", type=int, default=151, help="Moving window ocean clutter size in pixels")
    args = parser.parse_args()

    if not HAS_RASTERIO:
        print("Error: rasterio is required to run standalone segmentation.")
        sys.exit(1)

    print(f"Loading scene: {args.scene}")
    with rasterio.open(args.scene) as src:
        sar_arr = src.read()
        meta = {
            'width': src.width,
            'height': src.height,
            'crs': src.crs,
            'transform': src.transform,
            'pixel_size_x': src.res[0],
            'pixel_size_y': src.res[1],
            'path': args.scene
        }

    res = segment_candidate_dark_spots(
        sar_arr, meta,
        min_damping_db=args.min_damping,
        max_backscatter_db=args.max_backscatter,
        min_area_px=args.min_area_px,
        window_size=args.window_size
    )

    print(f"Extraction complete: {res.candidate_count} candidates found, total area: {res.total_candidate_area_km2:.4f} km²")

    out_dir = args.output or os.path.dirname(os.path.abspath(args.scene))
    base = os.path.splitext(os.path.basename(args.scene))[0]
    out_geojson = os.path.join(out_dir, f"{base}_candidates.geojson")
    export_candidate_geojson(res.candidates, meta, out_geojson)
    print(f"Exported candidates GeoJSON -> {out_geojson}")
