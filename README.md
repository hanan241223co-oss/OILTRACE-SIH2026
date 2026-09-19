# OILTRACE Satellite Marine Oil Spill Detection and Attribution

Welcome to OILTRACE, an automated marine surveillance system built to identify mineral oil spills in Sentinel-1 Synthetic Aperture Radar (SAR) imagery while suppressing natural lookalikes such as low wind pools, grease slicks, and vessel radar shadows.

OILTRACE uses a two stage Object Based Image Analysis (OBIA) detection pipeline. Rather than analyzing noisy pixels in isolation, the system evaluates connected dark spots with their surrounding ocean clutter, spatial geometry, and polarimetric characteristics.

---

## Architecture Overview

Traditional pixel classifiers produce thousands of false alarms in SAR scenes due to low wind speed areas and ship shadows. OILTRACE solves this by executing a sequential four stage object detection cascade:

```
[ Dual Polarization SAR GeoTIFF (VV / VH dB) ]
                      │
                      ▼
[ Stage 1: Autonomous Vessel & Radar Shadow Suppression ]
  • CFAR detector extracts metallic vessel targets
  • Directional shadow wedges mask non spill dark patches
                      │
                      ▼
[ Stage 2: Adaptive Ocean Clutter Candidate Segmentation ]
  • Moving window clutter estimation determines local damping contrast
  • 8-connectivity grouping extracts coherent dark spot polygons
                      │
                      ▼
[ Stage 3: 17-Dimensional Physical & Geometric Feature Extraction ]
  • Geodesic area, perimeter, and shape complexity
  • Backscatter damping ratio and Sobel perimeter edge sharpness
  • Spatial proximity to suspect vessels
                      │
                      ▼
[ Stage 4: Calibrated Random Forest Object Classification ]
  • Calibrated probability thresholding suppresses lookalikes
  • Confirmed slicks export to high resolution masks and GeoJSON polygons
```

Each stage is thoroughly documented with architectural decisions and mathematical derivations in the [docs/specs/](file:///teamspace/studios/this_studio/OILTRACE-SIH2026/docs/specs/) directory.

---

## Directory Structure

```
OILTRACE-SIH2026/
├── pipeline/                      # Standalone operational detection package
│   ├── detect_oil.py              # Unified CLI runner and detection cascade
│   ├── vessel_masking.py          # Stage 1: Vessel and radar shadow masking
│   ├── segmentation.py            # Stage 2: Adaptive ocean clutter candidate segmentation
│   ├── feature_extractor.py       # Stage 3: 17-dimensional object feature extraction
│   ├── obia_classifier.py         # Stage 4: Calibrated Random Forest inference engine
│   ├── data/
│   │   └── scene.tif              # Reference Sentinel-1 SAR scene (2048 x 2048)
│   ├── models/
│   │   └── best_obia_rf.joblib    # Pretrained OBIA model bundle with calibrated threshold
│   └── outputs/                   # Standard detection deliverables directory
│       ├── mask.png               # Binary detection mask (confirmed slicks = 255)
│       ├── scene_spills.geojson   # Georeferenced confirmed oil spill polygons
│       ├── scene_candidates.geojson # Segmented candidate polygons tagged with status
│       └── result.json            # Machine readable execution summary
├── docs/                          # Technical specifications and documentation
│   ├── specs/                     # Formal architecture specs for each stage
│   │   ├── 0001-autonomous-vessel-masking/
│   │   ├── 0002-adaptive-candidate-segmentation/
│   │   ├── 0003-physical-geometric-feature-extractor/
│   │   ├── 0004-object-level-random-forest-classifier/
│   │   └── 0005-two-stage-pipeline-verification/
│   ├── scope/scope.md             # Project roadmap and feature delivery scope
│   └── reviews/                   # Pre-merge code review reports
├── scripts/                       # Supplementary tools and surveillance dashboard
│   ├── run_detection_pipeline.py  # Top level master pipeline wrapper
│   ├── dashboard/                 # Interactive Streamlit surveillance dashboard
│   └── attribution/               # Vessel trajectory and drift hindcasting
└── requirements.txt               # Python package dependencies
```

---

## Installation

Ensure you have Python 3.10 or higher installed. Clone the repository and install the dependencies:

```bash
git clone https://github.com/hanan241223co-oss/OILTRACE-SIH2026.git
cd OILTRACE-SIH2026
pip install -r requirements.txt
```

---

## Quick Start

### 1. Run Detection on the Included Sample Scene

You can execute the entire two stage pipeline on the bundled Sentinel-1 SAR scene with one command:

```bash
python pipeline/detect_oil.py
```

This runs the detection cascade using [pipeline/data/scene.tif](file:///teamspace/studios/this_studio/OILTRACE-SIH2026/pipeline/data/scene.tif) and [pipeline/models/best_obia_rf.joblib](file:///teamspace/studios/this_studio/OILTRACE-SIH2026/pipeline/models/best_obia_rf.joblib), saving all deliverables directly into [pipeline/outputs/](file:///teamspace/studios/this_studio/OILTRACE-SIH2026/pipeline/outputs/).

### 2. Run Detection on Your Own GeoTIFF Scene

To run detection on any custom dual polarization Sentinel-1 SAR GeoTIFF file:

```bash
python pipeline/detect_oil.py --input /path/to/your_scene.tif --output /path/to/output_dir/
```

### 3. Run via Master Script

You can also launch detection using the master script:

```bash
python scripts/run_detection_pipeline.py --input pipeline/data/scene.tif --output_dir pipeline/outputs/
```

---

## Output Deliverables

When execution completes, you will find four standard deliverables in your output directory:

1. **`mask.png`**
   A high resolution binary raster mask matching the exact dimensions of your SAR scene. Pixels classified as genuine mineral oil spills receive a value of 255 (white), while ocean background and suppressed lookalikes receive 0 (black).

2. **`scene_spills.geojson`**
   A standard GeoJSON FeatureCollection containing polygons for all confirmed oil slicks in EPSG:4326 coordinates. Each polygon feature includes rich physical attributes:
   * `area_km2`: Geodesic surface area calculated via WGS84 ellipsoid
   * `confidence`: Calibrated spill probability score
   * `local_damping_db`: Radar contrast damping relative to surrounding sea clutter
   * `boundary_gradient_mean`: Sobel edge sharpness along the slick boundary
   * `distance_to_nearest_vessel_m`: Proximity to closest metallic vessel target

3. **`result.json`**
   A structured summary recording execution time, vessel counts, candidate counts, and confirmed slick totals:
   ```json
   {
     "scene_id": "data",
     "pipeline_mode": "two_stage",
     "model_path": ".../best_obia_rf.joblib",
     "decision_threshold": 0.6,
     "execution_time_seconds": 25.68,
     "vessel_masking": {
       "vessels_detected": 124,
       "shadow_pixels_masked": 0
     },
     "candidate_segmentation": {
       "total_candidates": 3576,
       "total_candidate_area_km2": 54.35
     },
     "object_classification": {
       "confirmed_slicks": 1,
       "suppressed_lookalikes": 3575,
       "confirmed_area_km2": 2.36
     }
   }
   ```

4. **`scene_candidates.geojson`**
   An inspection layer containing every segmented dark spot polygon tagged with its classification status (`confirmed_slick` or `lookalike_suppressed`) and estimated oil probability.

5. **`scene_detection_result.png`**
   A comprehensive multi-panel analytical surveillance dashboard presenting the dual polarization SAR scene, probability heatmap, binary mask, vector bounding boxes, close-up zoom on the primary slick, and detailed operational dossier scorecard.

6. **`scene_mask_overlay.png`**
   A high resolution visual composite rendering translucent crimson oil slicks with glowing edges directly overlaid on the Sentinel-1 radar backscatter.

---

## CLI Options & Parameter Tuning

The pipeline provides flexible command line switches to tune detection parameters:

| CLI Option | Default | Description |
|---|---|---|
| `--input` | `pipeline/data/scene.tif` | Path to dual polarization Sentinel-1 GeoTIFF (Band 0: VV, Band 1: VH in dB) |
| `--output` | `pipeline/outputs/` | Target directory where masks and GeoJSON deliverables are saved |
| `--mode` | `two_stage` | Detection mode: `two_stage` (OBIA Random Forest), `unet`, or `rf_pixel` |
| `--threshold` | Model bundle value | Probability cutoff for confirming mineral oil slicks (default calibrated at 0.60) |
| `--min_damping` | `2.2` dB | Minimum local ocean damping contrast required for candidate segmentation |
| `--max_backscatter` | `-20.0` dB | Maximum VV backscatter ceiling for candidate dark spot cores |
| `--window_size` | `151` px | Moving window kernel dimension in pixels used for ambient sea clutter estimation |
| `--min_px` | `50` px | Minimum pixel count required for a confirmed slick |
| `--no-vessel-masking` | `False` | Disables metallic vessel target detection and shadow corridor suppression |
| `--no-candidates` | `False` | Skips exporting the intermediate candidate polygon GeoJSON layer |

---

## Interactive Surveillance Dashboard

To explore detection results visually alongside maritime vessel tracking, launch the built-in Streamlit web application:

```bash
streamlit run scripts/dashboard/app.py
```

The web dashboard allows operators to:
* View Sentinel-1 SAR imagery with interactive polygon overlays
* Inspect candidate feature distributions and confidence scores
* Track vessel positions and simulate 12 hour spill drift trajectories

---

## Technical Specifications & Documentation

For comprehensive technical rationale, validation metrics, and mathematical formulas, explore the documentation in [docs/](file:///teamspace/studios/this_studio/OILTRACE-SIH2026/docs/):

* [0001 Autonomous Vessel Masking Spec](file:///teamspace/studios/this_studio/OILTRACE-SIH2026/docs/specs/0001-autonomous-vessel-masking/index.md): Ship reflector extraction and geometric shadow corridor projection
* [0002 Adaptive Candidate Segmentation Spec](file:///teamspace/studios/this_studio/OILTRACE-SIH2026/docs/specs/0002-adaptive-candidate-segmentation/index.md): Moving window ocean background contrast damping algorithm
* [0003 Physical Geometric Feature Extractor Spec](file:///teamspace/studios/this_studio/OILTRACE-SIH2026/docs/specs/0003-physical-geometric-feature-extractor/index.md): 17 dimensional polygon and polarimetric descriptor formulation
* [0004 Object Level Random Forest Classifier Spec](file:///teamspace/studios/this_studio/OILTRACE-SIH2026/docs/specs/0004-object-level-random-forest-classifier/index.md): Classifier training, cross validation, and probability calibration
* [0005 Two Stage Pipeline Verification Spec](file:///teamspace/studios/this_studio/OILTRACE-SIH2026/docs/specs/0005-two-stage-pipeline-verification/index.md): Unified CLI orchestration, deliverable verification, and zero candidate handling
* [Project Scope & Roadmap](file:///teamspace/studios/this_studio/OILTRACE-SIH2026/docs/scope/scope.md): Complete feature status across detection, attribution, and visualization slices
