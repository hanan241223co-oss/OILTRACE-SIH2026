# 0002. Adaptive Dark Spot Candidate Segmentation

**Date**: 2026-09-19
**Status**: Accepted

## Summary

This specification establishes an adaptive dark spot segmentation engine for synthetic aperture radar imagery. It extracts connected candidate oil spill polygons by calculating local ocean background clutter across a moving spatial window and isolating regions with strong radar damping contrast. The extracted candidate polygons form the discrete spatial inputs required for object level feature extraction and classification in subsequent pipeline stages.

## Requirements

**User stories**:
* As a marine surveillance operator, I want raw radar scenes segmented into discrete candidate dark spot polygons so that potential slicks can be systematically evaluated with object level features rather than noisy individual pixels.
* As an automated detection pipeline, I want vessel radar shadows and bright metal reflections excluded from candidate extraction so that ship artifacts do not trigger candidate spill alarms.

**Acceptance criteria**:

* **AC-1**: Moving window background clutter calculation. The system computes a local ambient ocean mean backscatter map across a configurable moving window (default 101 by 101 pixels) using box convolution normalized exclusively over valid ocean pixels. If a local window contains fewer than 100 valid ocean pixels (for example near coastlines or scene boundaries), the calculation smoothly falls back to the scene global ocean median backscatter.
* **AC-2**: Relative damping thresholding. The system isolates dark candidate pixels satisfying both a minimum local damping contrast ($DR = \bar{\sigma}^0_{\text{local}} - \sigma^0_{\text{pixel}} \ge 2.5\text{ dB}$) and an absolute backscatter ceiling ($\sigma^0_{VV} \le -20.0\text{ dB}$). Invalid or non ocean pixels in the damping ratio map are assigned zero.
* **AC-3**: Morphological closing and connected component labeling. The system applies a 3 by 3 morphological closing kernel to bridge minor interior gaps, labels connected components using 8-connectivity (full 3 by 3 neighborhood), prunes isolated speckle clusters smaller than 30 pixels (approximately 0.003 square kilometers), and re-indexes surviving candidate regions consecutively from 1 to N.
* **AC-4**: Vessel masking and proximity linkage. The system executes initial candidate extraction first, then passes the preliminary candidate mask to `vessel_masking.mask_vessel_shadows` from Feature 8. This ensures genuine discharging slicks ($DR \ge 2.5\text{ dB}$, boundary Sobel $\ge 12.0$, area $\ge 50$ px) are preserved while unverified radar shadow corridors are suppressed. Candidates within 35 pixels of a suspect vessel are tagged with `touches_suspect_vessel = True`, and geodesic distance to the closest vessel is recorded.
* **AC-5**: Spatial polygonization and structured export. The system extracts vector polygon boundaries mapped to geographic coordinates (WGS84 EPSG 4326) via the scene affine transform, calculates geodesic areas in square kilometers using `pyproj.Geod(ellps='WGS84')`, packages results into a structured container, and exports a valid GeoJSON FeatureCollection.

## Feature design

**Data model sketch**:

```python
@dataclass
class CandidateDarkSpot:
    candidate_id: str                      # Unique identifier (example spot_001)
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
    nearest_vessel_id: str | None          # Identifier of nearest vessel target if detected
    distance_to_nearest_vessel_m: float | None # Distance in meters to closest detected vessel
    touches_suspect_vessel: bool           # True if candidate boundary is within 35 px of a suspect discharging vessel
    is_confirmed_slick: bool | None = None # Set by Stage 2 classifier
    slick_probability: float | None = None # Set by Stage 2 classifier

@dataclass
class SegmentationResult:
    scene_id: str                          # Name of input scene
    candidate_count: int                   # Total candidate dark spots extracted
    total_candidate_area_km2: float        # Sum of candidate geodesic areas
    candidate_mask: np.ndarray             # 2D integer array of labeled candidate spots (0 = ocean, 1..N = spot)
    damping_ratio_map: np.ndarray          # 2D float32 array of relative damping contrast values (0.0 for invalid)
    candidates: list[CandidateDarkSpot]     # Extracted candidate objects sorted by area descending
```

**State transitions**:
Stateless batch execution. Functions accept scene rasters and return immutable dataclass containers.

**API surface**:

| Function | Module | Key inputs | Key outputs | Description | Key errors |
|---|---|---|---|---|---|
| `compute_adaptive_ocean_background` | `testing/segmentation.py` | `vv_db`: ndarray, `valid_mask`: ndarray, `window_size`: int (opt, default 101), `min_samples`: int (opt, default 100) | `local_ocean_mean`: ndarray (float32) | Moving window ocean mean backscatter via masked box convolution with global median fallback | Empty valid mask across entire scene |
| `compute_damping_ratio_map` | `testing/segmentation.py` | `vv_db`: ndarray, `local_ocean_mean`: ndarray, `valid_mask`: ndarray | `damping_map`: ndarray (float32) | Pixel level damping ratio in decibels, masked to 0.0 outside valid ocean | Shape mismatch |
| `segment_candidate_dark_spots` | `testing/segmentation.py` | `sar_data`: ndarray, `metadata`: dict, `min_damping_db`: float (opt, 2.5), `max_backscatter_db`: float (opt, -20.0), `min_area_px`: int (opt, 30), `window_size`: int (opt, 101), `vessels`: list (opt) | `result`: SegmentationResult | Extracts connected candidate dark spots, applies vessel masking, and calculates spatial metrics | Invalid raster dimensions |
| `export_candidate_geojson` | `testing/segmentation.py` | `candidates`: list[CandidateDarkSpot], `metadata`: dict, `output_path`: str | None | Writes candidate vector polygons to GeoJSON FeatureCollection | Directory unwritable |

**Value sourcing**:

| Action | Value produced or displayed | Source |
|---|---|---|
| Compute ocean background | `local_ocean_mean` | Derived from `vv_db` convolution normalized by `valid_mask` count, fallback to global median if count $< 100$ |
| Compute damping map | `damping_ratio_map` | Derived as `np.where(valid_mask, local_ocean_mean - vv_db, 0.0)` |
| Preliminary thresholding | Initial candidate mask | Derived from `(damping >= min_damping_db) & (vv <= max_backscatter_db) & valid_mask` |
| Vessel shadow integration | Cleaned candidate mask | Derived by running `mask_vessel_shadows` with preliminary candidate mask to preserve true slicks |
| Extract components | `candidate_mask`, `pixel_count`, `bbox_px` | Derived from `scipy.ndimage.label` with 8-connectivity (`np.ones((3, 3))`) after 3x3 binary closing |
| Calculate geodesic area | `area_km2` | Derived using `pyproj.Geod(ellps='WGS84').geometry_area_perimeter(geom)[0] / 1e6` |
| Calculate geographic coordinates | `bbox_wgs84`, `centroid_wgs84`, `geometry_wgs84` | Derived from `metadata['transform']`, `metadata['crs']`, and `rasterio.features.shapes` |
| Link vessel context | `nearest_vessel_id`, `distance_to_nearest_vessel_m`, `touches_suspect_vessel` | Derived from spatial query against `vessels` list from Feature 8 (touching within 35 px of suspect ship) |

**Key invariants**:
* Valid ocean mask excludes zero padding, nodata values, and land boundaries.
* Local ocean background calculation divides only by valid pixel counts, falling back to global ocean median if local valid count is under 100 pixels.
* Non ocean and invalid pixels in `damping_ratio_map` are strictly set to 0.0.
* Candidate components are labeled using 8-connectivity. Pruned components with fewer than `min_area_px` pixels are removed, and surviving components are renumbered consecutively from 1 to N.
* Candidate dark spot labels in `candidate_mask` match the `label_id` field of the corresponding `CandidateDarkSpot` entity.
* If zero candidates are found in a clean ocean scene, `candidate_count` is zero, `candidates` is an empty list, `candidate_mask` is an all zero array, and export produces a valid empty GeoJSON collection without exceptions.

**Security model**:
Local file system operations only. Output paths are created within the caller specified directory. No network sockets or external credentials required.

**Configuration required**:
CLI flags added to `testing/detect_oil.py`:
* `--min_damping`: float, minimum damping contrast threshold in decibels (default 2.5)
* `--max_backscatter`: float, maximum absolute VV backscatter ceiling in decibels (default -20.0)
* `--window_size`: int, local ocean moving window side length in pixels (default 101)
* `--min_candidate_px`: int, minimum candidate cluster size in pixels (default 30)
* `--no_candidate_segmentation`: flag, bypass candidate extraction if running legacy pixel baseline directly

**Critical test scenarios**:
* Happy path: Full candidate extraction on test scene `det_20260917163850_004aae/scene.tif`, correctly extracting candidate dark spot polygons, verifies **AC-1**, **AC-2**, **AC-3**, **AC-5**.
* Vessel shadow exclusion and slick preservation: Candidate extraction passes preliminary mask to vessel masking, preserving genuine slicks while suppressing radar shadow corridors, verifies **AC-4**.
* Discharging vessel linkage: A candidate dark spot connected to a suspect vessel within 35 pixels is flagged with `touches_suspect_vessel = True`, verifies **AC-4**.
* Clean ocean empty scene: Synthetic scene with uniform sea clutter produces zero candidate dark spots without errors, verifies **AC-1**, **AC-2**, **AC-5**.
* Border and coastal safety: Image with large nodata edge borders and dry land boundaries computes background clutter using global median fallback without edge artifacts, verifies **AC-1**, **AC-2**.

## Build plan

Following the Tracer Bullet build approach, candidate dark spot segmentation is constructed as a self contained module in `testing/` first, verified on real SAR data, and then wired into the detection pipeline:

* [x] 1. Implement `compute_adaptive_ocean_background` with masked uniform filtering and global median fallback, and `compute_damping_ratio_map` in `testing/segmentation.py`, satisfies **AC-1**, **AC-2**
* [x] 2. Implement morphological closing, 8-connectivity connected component labeling, area pruning, and consecutive re-indexing in `testing/segmentation.py`, satisfies **AC-3**
* [x] 3. Implement spatial polygonization, geographic coordinate projection, and geodesic area calculation via pyproj and rasterio shapes in `testing/segmentation.py`, satisfies **AC-5**
* [x] 4. Implement preliminary mask pass to vessel masking, radar shadow suppression, and suspect ship proximity linkage in `testing/segmentation.py`, satisfies **AC-4**
* [x] 5. Integrate candidate segmentation into `testing/detect_oil.py` as an operational stage with CLI flags and automated candidate GeoJSON export, satisfies **AC-4**, **AC-5**

## Consequences

**Positive**:
* Eliminates reliance on brittle global decibel thresholds across variable sea states.
* Transforms continuous pixel arrays into clean spatial objects required for OBIA random forest classification.
* Suppresses single pixel radar speckle while preserving contiguous oil slick structures.
* Solves coupling with Feature 8: preliminary dark spots allow vessel masking to preserve discharging slicks while purging shadows.
* Adheres to root `AGENTS.md` standard using geodesic area calculations in square kilometers.

**Negative / tradeoffs**:
* Moving window convolution adds approximately 0.2 to 0.5 seconds of processing time per scene.
* Very large low wind pools will be admitted into the candidate list, relying on Stage 2 object features to reject them.

**Neutral**:
* Generates an intermediate `{base_name}_candidates.geojson` vector deliverable alongside the final detection output.
