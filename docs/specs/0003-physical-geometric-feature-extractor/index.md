# 0003. Physical & Geometric Object Feature Extractor

**Date**: 2026-09-19
**Status**: Accepted

## Summary

This specification establishes a physical and geometric object feature extraction engine for candidate radar dark spots. It analyzes each candidate polygon segmented from synthetic aperture radar scenes, computing 17 specialized shape, backscatter, boundary gradient, and spatial proximity metrics. These extracted feature vectors provide the discriminative inputs needed by object level machine learning classifiers to distinguish genuine mineral oil slicks from natural ocean lookalikes.

## Requirements

**User stories**:
* As a surveillance data scientist, I want candidate dark spot polygons enriched with geometric shape, physical radar contrast, and edge sharpness metrics so that machine learning classifiers can separate true oil spills from natural ocean lookalikes.
* As an automated pipeline developer, I want extracted features packaged into structured typed dataclasses and exportable as tabular matrices so that models can be trained and evaluated using standard scikit-learn interfaces.

**Acceptance criteria**:

* **AC-1**: Geometric morphology and shape descriptor extraction. The system computes 7 geometric descriptors for every candidate polygon: geodesic area (`area_km2`), geodesic perimeter (`perimeter_km`), elongation aspect ratio ($\ge 1.0$) via the minimum bounding rotated rectangle, circularity compactness ($4\pi A / P^2 \le 1.0$), perimeter area complexity ratio ($P / \sqrt{A}$), perimeter area fractal dimension ($2 \ln(P_{px}/4) / \ln(A_{px})$ computed in raster pixel units and clamped to $[1.0, 2.0]$), and bounding box area occupancy extent. Geodesic area in square kilometers and perimeter in kilometers are computed using the WGS84 ellipsoid.
* **AC-2**: Physical radar backscatter and polarimetric contrast extraction. The system extracts radar backscatter statistics inside each candidate polygon across VV and VH decibel channels: mean VV decibels, mean VH decibels, peak dark core minimum VV decibels, polarimetric difference ratio (mean VV decibels minus mean VH decibels), inner slick standard deviation (texture homogeneity, set to 0.0 for candidates under 2 pixels), and the local damping ratio contrast ($DR = \bar{\sigma}^0_{\text{local}} - \bar{\sigma}^0_{\text{slick}}$) relative to the surrounding ambient ocean clutter map from Feature 9.
* **AC-3**: Interfacial boundary gradient sharpness calculation. The system computes the boundary transition sharpness of candidate slicks by precomputing the full scene 3 by 3 Sobel gradient magnitude on the VV decibel channel, sampling gradient values along a 3 pixel outer dilated perimeter boundary ring masked strictly by the ocean `valid_mask` to prevent false nodata edge steps, and extracting both the mean boundary gradient and peak 95th percentile boundary gradient.
* **AC-4**: Spatial contextual vessel proximity features. The system derives vessel proximity metrics from Feature 8 and Feature 9: distance to the nearest detected vessel in meters (imputed to 10,000 meters when no vessels are present in the scene) and a binary indicator flag `touches_suspect_vessel` (1.0 if the candidate boundary is within 35 pixels of a suspect discharging vessel, else 0.0).
* **AC-5**: Dual container encapsulation and tabular matrix export. The system packages extracted features into typed `CandidateObjectFeatures` dataclass entities attached to each `CandidateDarkSpot`, constructs a structured 17-feature dictionary `feature_vector` per spot with canonical ordering, and provides an export function converting candidate collections into clean NumPy arrays (shape $[N, 17]$) and pandas DataFrames with verified feature column headers for downstream machine learning training and inference.

## Feature design

**Canonical feature vector (17 features)**:

```python
CANONICAL_FEATURE_NAMES = [
    # Geometric shape morphology (7 features)
    "area_km2",                     # 1. Geodesic area in square kilometers
    "perimeter_km",                # 2. Geodesic perimeter in kilometers
    "elongation",                  # 3. Major axis / minor axis of minimum rotated rectangle (>= 1.0)
    "circularity",                 # 4. 4 * pi * area / perimeter^2 (in (0.0, 1.0])
    "complexity_ratio",            # 5. perimeter / sqrt(area)
    "fractal_dimension",           # 6. 2 * ln(P_px / 4) / ln(A_px) clamped to [1.0, 2.0]
    "bbox_extent",                 # 7. candidate pixel area / bounding box pixel area
    
    # Physical radar backscatter (6 features)
    "mean_vv_db",                  # 8. Mean VV decibels inside candidate
    "mean_vh_db",                  # 9. Mean VH decibels inside candidate
    "min_vv_db",                   # 10. Minimum VV decibels (peak dark core)
    "vv_vh_ratio",                 # 11. Polarimetric difference: mean_vv_db - mean_vh_db
    "std_vv_db",                   # 12. Standard deviation of VV decibels inside slick
    "local_damping_db",            # 13. Ambient ocean clutter mean minus slick mean VV
    
    # Interfacial boundary gradient (2 features)
    "boundary_gradient_mean",      # 14. Mean Sobel gradient magnitude on valid 3 px perimeter ring
    "boundary_gradient_max",       # 15. 95th percentile Sobel gradient on valid perimeter ring
    
    # Spatial contextual proximity (2 features)
    "distance_to_nearest_vessel_m",# 16. Distance to closest vessel (10000.0 if no vessels)
    "touches_suspect_vessel",      # 17. 1.0 if candidate is within 35 px of suspect ship, else 0.0
]
```

**Data model sketch**:

```python
@dataclass
class CandidateObjectFeatures:
    candidate_id: str                          # Unique identifier matching CandidateDarkSpot (e.g. spot_001)
    
    # 1. Geometric shape morphology (7 features)
    area_km2: float                            # Geodesic area in square kilometers
    perimeter_km: float                        # Geodesic perimeter in kilometers
    elongation: float                          # Aspect ratio of minimum bounding rotated box (major / minor >= 1.0)
    circularity: float                         # 4 * pi * area / perimeter^2 (compactness metric in (0.0, 1.0])
    complexity_ratio: float                    # perimeter / sqrt(area)
    fractal_dimension: float                   # 2 * ln(P_px / 4) / ln(A_px) bounded in [1.0, 2.0]
    bbox_extent: float                         # Ratio of candidate pixel area to bounding box pixel area
    
    # 2. Physical radar backscatter (6 features)
    mean_vv_db: float                          # Mean VV backscatter inside slick in decibels
    mean_vh_db: float                          # Mean VH backscatter inside slick in decibels
    min_vv_db: float                           # Minimum VV backscatter (peak dark core) in decibels
    vv_vh_ratio: float                         # Polarimetric difference: mean_vv_db - mean_vh_db
    std_vv_db: float                           # Standard deviation of VV backscatter (slick texture)
    local_damping_db: float                    # Relative damping contrast against local ocean background in dB
    
    # 3. Interfacial boundary gradient (2 features)
    boundary_gradient_mean: float              # Mean Sobel gradient magnitude on valid 3 px perimeter ring
    boundary_gradient_max: float               # 95th percentile Sobel gradient magnitude on boundary
    
    # 4. Spatial contextual proximity (2 features)
    distance_to_nearest_vessel_m: float        # Distance to closest vessel in meters (imputed to 10000.0 if none)
    touches_suspect_vessel: float              # 1.0 if within 35 px of suspect discharging vessel, else 0.0
    
    # Structured export dictionary
    feature_vector: dict[str, float]           # 17-key dictionary following CANONICAL_FEATURE_NAMES order

@dataclass
class FeatureExtractionResult:
    scene_id: str                              # Scene identifier
    candidate_count: int                       # Total candidates processed
    feature_names: list[str]                   # Ordered list of 17 canonical feature keys
    feature_matrix: np.ndarray                 # 2D float32 array of shape (candidate_count, 17)
    candidate_features: list[CandidateObjectFeatures] # List of extracted feature records
```

**State transitions**:
Stateless batch execution. Functions take candidate polygons, candidate masks, scene rasters, and ocean clutter maps, returning immutable dataclass containers.

**API surface**:

| Function | Module | Key inputs | Key outputs | Description | Key errors |
|---|---|---|---|---|---|
| `compute_polygon_geometric_features` | `testing/feature_extractor.py` | `contour_px`: list, `pixel_count`: int, `bbox_px`: tuple, `area_km2`: float, `perimeter_km`: float | `dict[str, float]` | Calculates elongation, circularity, complexity ratio, pixel fractal dimension, and extent | Fewer than 3 contour points |
| `compute_boundary_gradient_sharpness` | `testing/feature_extractor.py` | `spot_slice_mask`: ndarray, `sobel_mag_slice`: ndarray, `valid_slice`: ndarray | `tuple[float, float]` | Computes mean and 95th percentile gradient along 3 px outer dilated ring masked by ocean validity | Empty slice mask |
| `extract_candidate_features` | `testing/feature_extractor.py` | `candidates`: list[CandidateDarkSpot], `candidate_mask`: ndarray, `sar_data`: ndarray, `damping_map`: ndarray, `metadata`: dict, `vessels`: list (opt) | `FeatureExtractionResult` | Extracts full 17 feature vectors for all candidate dark spots in a scene | Invalid raster dimensions |
| `export_features_tabular` | `testing/feature_extractor.py` | `result`: FeatureExtractionResult, `output_csv_path`: str (opt) | `pandas.DataFrame` | Converts extraction results into a pandas DataFrame with canonical columns | Empty candidate list |

**Value sourcing**:

| Action | Value produced or displayed | Source |
|---|---|---|
| Extract shape geometry | `elongation`, `bbox_extent` | Derived via `shapely.geometry.Polygon(contour).minimum_rotated_rectangle` with minor axis capped at minimum 1e-3 |
| Extract complexity | `circularity`, `complexity_ratio` | Derived from geodesic `area_km2` and `perimeter_km` via pyproj |
| Extract fractal dimension | `fractal_dimension` | Derived as $2 \ln(P_{px}/4) / \ln(A_{px})$ from pixel perimeter and area, clamped to $[1.0, 2.0]$ |
| Re-use candidate backscatter | `mean_vv_db`, `mean_vh_db`, `min_vv_db`, `local_damping_db` | Reused directly from precomputed attributes in `CandidateDarkSpot` |
| Compute texture & polarimetry | `std_vv_db`, `vv_vh_ratio` | Derived from pixel values inside candidate mask indexed into `sar_data` (default std 0.0 if px < 2) |
| Compute boundary sharpness | `boundary_gradient_mean`, `boundary_gradient_max` | Derived from precomputed scene `scipy.ndimage.sobel` magnitude sampled over 3 px outer dilated ring masked by `valid_mask` |
| Re-use vessel context | `distance_to_nearest_vessel_m`, `touches_suspect_vessel` | Reused from `CandidateDarkSpot` (imputed to 10000.0 m and 0.0 touches if no vessels) |
| Assemble matrix | `feature_matrix` | Derived by stacking numerical `feature_vector` rows into 2D float32 array of shape $(N, 17)$ |

**Key invariants**:
* The feature vector always contains exactly 17 defined numerical features in the strict order defined by `CANONICAL_FEATURE_NAMES`.
* No feature value may be NaN or Inf. If a degenerate candidate has zero area or perimeter, safe defaults are assigned ($1.0$ for elongation, $0.0$ for circularity, complexity ratio, and boundary gradient, $1.0$ for fractal dimension).
* If a scene contains zero detected vessels, `distance_to_nearest_vessel_m` is imputed to 10,000.0 meters and `touches_suspect_vessel` is set to 0.0.
* Feature matrix shape is strictly $(N, 17)$ where $N$ equals `candidate_count`.
* If a clean scene produces zero candidates, `feature_matrix` has shape $(0, 17)$, `candidate_features` is empty, and DataFrame export produces an empty table with 17 column headers without errors.

**Security model**:
Local file system operations only. Output paths are created within caller specified directories. No network sockets or external credentials required.

**Configuration required**:
CLI flags added to `testing/detect_oil.py`:
* `--no_feature_extraction`: flag, disable object feature extraction on segmented candidate polygons
* `--features_csv`: str (opt), path to export extracted candidate feature table as CSV

**Critical test scenarios**:
* Happy path: Full feature extraction on test scene `det_20260917163850_004aae/scene.tif`, extracting 17 valid features per candidate polygon, verifies **AC-1**, **AC-2**, **AC-3**, **AC-4**, **AC-5**.
* Elongated vs circular candidate discrimination: Synthetic elongated slick (aspect ratio 8.0) produces high elongation and low circularity, while circular pool produces elongation near 1.0 and circularity near 1.0, verifies **AC-1**.
* Boundary sharpness discrimination: Sharp step transition slick boundary produces higher Sobel gradient than smooth blurred boundary, verifies **AC-3**.
* Empty candidate scene handling: Clean ocean scene with zero candidate polygons returns empty feature matrix of shape $(0, 17)$ without crashing, verifies **AC-5**.
* Missing vessel scene handling: Scene with no detected vessels correctly assigns 10,000 meters distance and 0.0 touches flag without NaN, verifies **AC-4**.

## Build plan

Following the Tracer Bullet build approach, feature extraction is implemented in `testing/` first, verified against real candidate polygons, and integrated into the detection pipeline:

* [x] 1. Add `features: Optional[CandidateObjectFeatures] = None` to `CandidateDarkSpot` in `testing/segmentation.py`, satisfies **AC-5**
* [x] 2. Implement geometric morphology extraction (`compute_polygon_geometric_features`) covering elongation, circularity, complexity ratio, fractal dimension, and extent in `testing/feature_extractor.py`, satisfies **AC-1**
* [x] 3. Implement radar backscatter and texture standard deviation extraction in `testing/feature_extractor.py`, satisfies **AC-2**
* [x] 4. Implement precomputed scene Sobel gradient magnitude and masked boundary ring sampling (`compute_boundary_gradient_sharpness`) in `testing/feature_extractor.py`, satisfies **AC-3**
* [x] 5. Implement dual container encapsulation (`CandidateObjectFeatures`, `FeatureExtractionResult`) and tabular matrix export (`export_features_tabular`) in `testing/feature_extractor.py`, satisfies **AC-4**, **AC-5**
* [x] 6. Integrate feature extraction into `testing/detect_oil.py` as Step 3c with automated CSV export, satisfies **AC-5**

## Consequences

**Positive**:
* Equips the pipeline with 17 physically grounded, interpretable object features.
* Eliminates reliance on single pixel values, enabling robust discrimination between mineral oil slicks and natural sea surface lookalikes.
* Precomputing the scene Sobel gradient once achieves $O(1)$ boundary sampling per candidate, completing feature extraction across 3,500 candidates in under 1 second.
* Provides a standard scikit-learn tabular format for training the Feature 11 Random Forest model.

**Negative / tradeoffs**:
* Dilated boundary Sobel sampling adds 0.2 seconds of processing time per scene.
* Imputing vessel distance to 10,000 meters when no vessels are present requires models to learn that large values mean absence of vessels.

**Neutral**:
* Adds an optional `{base_name}_features.csv` tabular deliverable alongside candidate GeoJSON polygons.
