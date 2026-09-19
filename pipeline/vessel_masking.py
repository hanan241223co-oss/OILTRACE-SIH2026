#!/usr/bin/env python3
"""
OILTRACE Autonomous Vessel and Bright Target Masking Module
------------------------------------------------------------
Identifies metallic ship corner reflectors directly from dual polarization
Sentinel-1 SAR backscatter scenes. Projects geometric radar shadow corridors
behind vessels and suppresses false alarm dark patches while preserving
genuine oil slicks through physical damping and edge gradient checks.
"""

import os
import math
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
from scipy import ndimage
from scipy.ndimage import uniform_filter, sobel, binary_dilation, label, find_objects
from shapely.geometry import Point, Polygon

try:
    import rasterio
    from rasterio.transform import xy
    from rasterio.warp import transform
    from rasterio.features import rasterize
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


@dataclass
class VesselTarget:
    """Represents a metallic vessel reflector detected in SAR backscatter."""
    target_id: str
    pixel_x: float
    pixel_y: float
    longitude: Optional[float]
    latitude: Optional[float]
    peak_vv_db: float
    peak_vh_db: float
    pixel_area: int
    is_suspect_source: bool = False


@dataclass
class ShadowCorridor:
    """Represents the projected geometric radar shadow corridor cast by a vessel."""
    corridor_id: str
    target_id: str
    geometry_type: str
    range_azimuth_deg: Optional[float]
    length_pixels: float
    width_pixels: float
    suppressed_pixel_count: int


def detect_vessel_targets(vv_db: np.ndarray,
                          vh_db: np.ndarray,
                          valid_mask: np.ndarray,
                          metadata: dict) -> List[VesselTarget]:
    """
    Identifies metallic vessel reflectors using vectorized dual polarization peak detection.
    Criteria: VV > -2.0 dB or VH > -12.0 dB with local contrast exceeding 10.0 dB
    above the 51x51 moving window local mean, capped to top 300 targets by peak VV.
    """
    if vv_db.shape != vh_db.shape:
        raise ValueError(f"VV shape {vv_db.shape} does not match VH shape {vh_db.shape}")

    # Compute moving window local mean across valid ocean area
    local_vv_mean = uniform_filter(vv_db, size=51)
    local_vh_mean = uniform_filter(vh_db, size=51)

    contrast_vv = vv_db - local_vv_mean
    contrast_vh = vh_db - local_vh_mean

    # Vectorized bright reflector peak condition
    is_peak = (vv_db > -2.0) | (vh_db > -12.0)
    has_contrast = (contrast_vv > 10.0) | (contrast_vh > 10.0)
    reflector_mask = valid_mask & is_peak & has_contrast

    labeled_reflectors, num_reflectors = label(reflector_mask)
    if num_reflectors == 0:
        return []

    slices = find_objects(labeled_reflectors)
    gt_transform = metadata.get('transform')
    crs = metadata.get('crs')

    candidate_vessels = []
    for idx, slc in enumerate(slices):
        if slc is None:
            continue
        comp_id = idx + 1
        comp_mask = (labeled_reflectors[slc] == comp_id)
        area = int(np.sum(comp_mask))
        if area == 0:
            continue

        r_slice, c_slice = slc
        rows_sub, cols_sub = np.where(comp_mask)
        rows = rows_sub + r_slice.start
        cols = cols_sub + c_slice.start

        cy_px = float(np.mean(rows))
        cx_px = float(np.mean(cols))

        peak_vv = float(np.max(vv_db[slc][comp_mask]))
        peak_vh = float(np.max(vh_db[slc][comp_mask]))

        # Transform raster pixel coordinates to geographic WGS84
        lon_val = None
        lat_val = None
        if gt_transform is not None and HAS_RASTERIO:
            try:
                map_x, map_y = xy(gt_transform, cy_px, cx_px, offset='center')
                if crs:
                    lons, lats = transform(crs, 'EPSG:4326', [map_x], [map_y])
                    lon_val = float(lons[0])
                    lat_val = float(lats[0])
                else:
                    lon_val = float(map_x)
                    lat_val = float(map_y)
            except Exception:
                pass

        candidate_vessels.append({
            'pixel_x': cx_px,
            'pixel_y': cy_px,
            'longitude': lon_val,
            'latitude': lat_val,
            'peak_vv_db': peak_vv,
            'peak_vh_db': peak_vh,
            'pixel_area': area
        })

    # Sort candidates by peak VV descending to retain the strongest reflectors
    candidate_vessels.sort(key=lambda v: v['peak_vv_db'], reverse=True)

    # Clutter cap: retain maximum 300 vessel targets to avoid port clutter explosion
    capped_vessels = candidate_vessels[:300]

    vessel_targets = []
    for i, v_info in enumerate(capped_vessels):
        target_id = f"vessel_{i + 1:03d}"
        vessel_targets.append(VesselTarget(
            target_id=target_id,
            pixel_x=v_info['pixel_x'],
            pixel_y=v_info['pixel_y'],
            longitude=v_info['longitude'],
            latitude=v_info['latitude'],
            peak_vv_db=v_info['peak_vv_db'],
            peak_vh_db=v_info['peak_vh_db'],
            pixel_area=v_info['pixel_area'],
            is_suspect_source=False
        ))

    return vessel_targets


def project_shadow_corridors(vessels: List[VesselTarget],
                             shape: Tuple[int, int],
                             candidate_mask: np.ndarray,
                             metadata: dict) -> Tuple[np.ndarray, List[ShadowCorridor]]:
    """
    Constructs directional radar shadow corridors projecting down-range from each vessel.
    Falls back to a 15-pixel radial dilation buffer when satellite orbit look direction is absent.
    """
    H, W = shape
    shadow_mask = np.zeros((H, W), dtype=bool)
    corridors = []

    if not vessels:
        return shadow_mask, corridors

    look_dir = metadata.get('look_direction')
    if look_dir is None:
        look_dir = metadata.get('range_azimuth_deg')

    use_directional = (look_dir is not None)
    geom_type = "wedge" if use_directional else "radial_buffer"
    length_pixels = 25.0 if use_directional else 15.0

    for i, v in enumerate(vessels):
        corridor_id = f"corridor_{i + 1:03d}"
        v_width = max(3.0, math.sqrt(v.pixel_area)) + 4.0

        if use_directional:
            theta = math.radians(float(look_dir))
            ux, uy = math.cos(theta), math.sin(theta)
            px, py = -math.sin(theta), math.cos(theta)
            hw = v_width / 2.0

            poly_pts = [
                (v.pixel_x + hw * px, v.pixel_y + hw * py),
                (v.pixel_x - hw * px, v.pixel_y - hw * py),
                (v.pixel_x + length_pixels * ux - hw * px, v.pixel_y + length_pixels * uy - hw * py),
                (v.pixel_x + length_pixels * ux + hw * px, v.pixel_y + length_pixels * uy + hw * py),
            ]
            geom = Polygon(poly_pts)
        else:
            geom = Point(v.pixel_x, v.pixel_y).buffer(15.0)

        # Rasterize geometry onto raster coordinate grid
        if HAS_RASTERIO:
            v_corridor_mask = rasterize([geom], out_shape=(H, W), fill=0, default_value=1, dtype=np.uint8).astype(bool)
        else:
            # Fallback circular rasterizer
            y_coords, x_coords = np.ogrid[:H, :W]
            dist_sq = (x_coords - v.pixel_x)**2 + (y_coords - v.pixel_y)**2
            v_corridor_mask = (dist_sq <= 15.0**2)

        suppressed_count = int(np.sum(v_corridor_mask & candidate_mask))
        shadow_mask |= v_corridor_mask

        corridors.append(ShadowCorridor(
            corridor_id=corridor_id,
            target_id=v.target_id,
            geometry_type=geom_type,
            range_azimuth_deg=float(look_dir) if use_directional else None,
            length_pixels=length_pixels,
            width_pixels=v_width,
            suppressed_pixel_count=suppressed_count
        ))

    return shadow_mask, corridors


def mask_vessel_shadows(sar_data: np.ndarray,
                        candidate_mask: np.ndarray,
                        metadata: dict,
                        output_geojson_path: Optional[str] = None) -> Tuple[np.ndarray, List[VesselTarget], Dict[str, Any]]:
    """
    Main entrypoint: detects vessels, computes shadow corridors, evaluates physical damping
    and edge sharpness contrast, suppresses shadow false alarms, preserves genuine slicks,
    and exports detected vessels as a GeoJSON FeatureCollection.
    """
    if sar_data.ndim == 3 and sar_data.shape[0] == 2:
        sar_data = np.transpose(sar_data, (1, 2, 0))

    vv_raw = sar_data[:, :, 0].astype(np.float32)
    vh_raw = sar_data[:, :, 1].astype(np.float32)
    H, W = vv_raw.shape

    valid_mask = ((vv_raw != 0.0) | (vh_raw != 0.0)) & ~np.isnan(vv_raw)
    vv_db = np.clip(vv_raw, -50.0, 5.0)
    vh_db = np.clip(vh_raw, -50.0, 5.0)

    # 1. Detect bright metallic reflectors
    vessels = detect_vessel_targets(vv_db, vh_db, valid_mask, metadata)

    if not vessels:
        geojson_empty = {
            "type": "FeatureCollection",
            "name": "detected_vessel_targets",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": []
        }
        if output_geojson_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_geojson_path)), exist_ok=True)
            with open(output_geojson_path, 'w', encoding='utf-8') as f:
                json.dump(geojson_empty, f, indent=2)
        return candidate_mask.copy(), vessels, geojson_empty

    # 2. Project shadow corridors
    shadow_mask, corridors = project_shadow_corridors(vessels, (H, W), candidate_mask, metadata)
    corridor_by_target = {c.target_id: c for c in corridors}

    # 3. Physical contrast check and slick preservation override
    # Compute Sobel edge gradient magnitude on VV backscatter
    sx = sobel(vv_db, axis=0)
    sy = sobel(vv_db, axis=1)
    grad_mag = np.hypot(sx, sy)

    cleaned_mask = candidate_mask.copy().astype(bool)
    cand_labeled, num_candidates = label(candidate_mask)

    if num_candidates > 0:
        cand_slices = find_objects(cand_labeled)
        for i, slc in enumerate(cand_slices):
            if slc is None:
                continue
            cand_id = i + 1

            # Bounding box coordinates with 25-pixel margin for ocean background ring
            ymin = max(0, slc[0].start - 25)
            ymax = min(H, slc[0].stop + 25)
            xmin = max(0, slc[1].start - 25)
            xmax = min(W, slc[1].stop + 25)

            sub_cand = (cand_labeled[ymin:ymax, xmin:xmax] == cand_id)
            if not np.any(sub_cand):
                continue

            sub_shadow = shadow_mask[ymin:ymax, xmin:xmax]
            # Check if this candidate polygon intersects any vessel shadow corridor
            if not np.any(sub_cand & sub_shadow):
                continue

            # Candidate intersects shadow corridor: evaluate physical properties
            cand_area = int(np.sum(sub_cand))
            sub_vv = vv_db[ymin:ymax, xmin:xmax]
            sub_valid = valid_mask[ymin:ymax, xmin:xmax]

            # Background ocean ring (between 4 and 20 pixels dilation)
            d20 = binary_dilation(sub_cand, iterations=20)
            d4 = binary_dilation(sub_cand, iterations=4)
            sub_bg = d20 & (~d4) & sub_valid

            mean_slick_vv = float(np.mean(sub_vv[sub_cand]))
            mean_bg_vv = float(np.mean(sub_vv[sub_bg])) if np.sum(sub_bg) > 0 else mean_slick_vv
            damping_ratio = mean_bg_vv - mean_slick_vv

            # Boundary Sobel gradient
            sub_border = binary_dilation(sub_cand, iterations=1) & (~sub_cand)
            sub_grad = grad_mag[ymin:ymax, xmin:xmax]
            boundary_grad_mean = float(np.mean(sub_grad[sub_border])) if np.sum(sub_border) > 0 else 0.0

            # True oil slick preservation check (AC-4 and Key Invariant)
            is_true_slick = (damping_ratio >= 2.5) and (boundary_grad_mean >= 12.0) and (cand_area >= 50)

            if is_true_slick:
                # Genuine oil spill touching a vessel: preserve slick and flag suspect source
                rows, cols = np.where(sub_cand)
                cand_center_y = float(np.mean(rows) + ymin)
                cand_center_x = float(np.mean(cols) + xmin)

                # Find nearest vessel within range
                for v in vessels:
                    dist = math.hypot(v.pixel_x - cand_center_x, v.pixel_y - cand_center_y)
                    if dist <= 35.0:
                        v.is_suspect_source = True
            else:
                # Unverified radar shadow void or low contrast wake: suppress from candidate mask
                full_cand_comp = (cand_labeled == cand_id)
                # Suppress the shadow portion of this candidate
                cleaned_mask[full_cand_comp & shadow_mask] = False

    # Recalculate exact suppressed pixel counts per corridor
    actually_suppressed = (candidate_mask.astype(bool) & ~cleaned_mask)
    for corr in corridors:
        # Tally suppressed pixels within this corridor
        v_target = next((v for v in vessels if v.target_id == corr.target_id), None)
        if v_target:
            if corr.geometry_type == "wedge" and corr.range_azimuth_deg is not None:
                theta = math.radians(float(corr.range_azimuth_deg))
                ux, uy = math.cos(theta), math.sin(theta)
                px, py = -math.sin(theta), math.cos(theta)
                hw = corr.width_pixels / 2.0
                poly_pts = [
                    (v_target.pixel_x + hw * px, v_target.pixel_y + hw * py),
                    (v_target.pixel_x - hw * px, v_target.pixel_y - hw * py),
                    (v_target.pixel_x + corr.length_pixels * ux - hw * px, v_target.pixel_y + corr.length_pixels * uy - hw * py),
                    (v_target.pixel_x + corr.length_pixels * ux + hw * px, v_target.pixel_y + corr.length_pixels * uy + hw * py),
                ]
                geom = Polygon(poly_pts)
            else:
                geom = Point(v_target.pixel_x, v_target.pixel_y).buffer(15.0)

            if HAS_RASTERIO:
                c_mask = rasterize([geom], out_shape=(H, W), fill=0, default_value=1, dtype=np.uint8).astype(bool)
            else:
                y_c, x_c = np.ogrid[:H, :W]
                c_mask = ((x_c - v_target.pixel_x)**2 + (y_c - v_target.pixel_y)**2 <= 15.0**2)
            corr.suppressed_pixel_count = int(np.sum(c_mask & actually_suppressed))

    # 4. Construct GeoJSON FeatureCollection of Point features
    features = []
    for v in vessels:
        if v.longitude is not None and v.latitude is not None:
            geom = {"type": "Point", "coordinates": [round(v.longitude, 6), round(v.latitude, 6)]}
        else:
            geom = {"type": "Point", "coordinates": [round(v.pixel_x, 2), round(v.pixel_y, 2)]}

        corr = corridor_by_target.get(v.target_id)
        suppressed_count = corr.suppressed_pixel_count if corr else 0

        properties = {
            "target_id": v.target_id,
            "pixel_x": round(v.pixel_x, 2),
            "pixel_y": round(v.pixel_y, 2),
            "peak_vv_db": round(v.peak_vv_db, 2),
            "peak_vh_db": round(v.peak_vh_db, 2),
            "pixel_area": int(v.pixel_area),
            "is_suspect_source": bool(v.is_suspect_source),
            "suppressed_shadow_pixels": int(suppressed_count)
        }
        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": properties
        })

    geojson_dict = {
        "type": "FeatureCollection",
        "name": "detected_vessel_targets",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features
    }

    if output_geojson_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_geojson_path)), exist_ok=True)
        with open(output_geojson_path, 'w', encoding='utf-8') as f:
            json.dump(geojson_dict, f, indent=2)

    return cleaned_mask.astype(np.uint8), vessels, geojson_dict
