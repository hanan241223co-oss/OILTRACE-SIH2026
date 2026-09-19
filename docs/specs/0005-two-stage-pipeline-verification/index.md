# 0005. Two Stage Pipeline Verification in Testing

**Date**: 2026-09-19
**Status**: Accepted

## Summary

This specification defines the integration and verification of the two stage object based detection pipeline in `testing/detect_oil.py`. It chains autonomous vessel shadow masking, adaptive ocean candidate segmentation, physical geometric feature extraction, and object level Random Forest classification into a unified command line tool. The pipeline delivers confirmed slick vector polygons, binary raster masks, and structured run summaries while maintaining full backward compatibility with existing deep learning U-Net and legacy pixel baseline models.

## Requirements

**User stories**:
* As a marine surveillance operator, I want a single command line interface that executes the complete two stage detection pipeline on Sentinel-1 SAR scenes, so that true oil slicks are detected and natural ocean lookalikes are suppressed automatically.
* As an automated pipeline developer, I want consistent deliverable files including GeoJSON vector polygons, binary raster masks, and JSON execution summaries, so that downstream attribution and dashboard modules can ingest verified detection results.

**Acceptance criteria**:
* **AC-1**: CLI interface and mode selection. The script `testing/detect_oil.py` provides a `--mode` argument accepting `two_stage`, `unet`, and `rf_pixel`. When `--mode` is omitted, the CLI automatically selects `two_stage` if `testing/best_obia_rf.joblib` exists, or falls back to `unet` if `testing/best_unet.pt` is present. Full backward compatibility is preserved for existing commands using `--model`, `--input`, `--output`, `--threshold`, `--no-vessel-masking`, `--no-candidates`, and `--device`.
* **AC-2**: End to end sequential detection cascade. When running in `two_stage` mode, the pipeline executes a sequential processing cascade: (1) ingests the dual polarization SAR GeoTIFF, (2) detects bright metallic vessel targets and generates directional radar shadow corridor masks unless disabled via `--no-vessel-masking`, (3) segments dark spot candidate polygons using ocean background damping contrast, (4) extracts 17 physical and geometric features per candidate polygon, and (5) evaluates candidates through `ObjectClassifier` to classify confirmed oil slicks versus suppressed lookalikes.
* **AC-3**: Operating decision thresholding and runtime override. Confirmed slicks are determined using the calibrated probability threshold stored in the model bundle (default 0.60). The CLI accepts an optional `--threshold` argument allowing runtime override of this decision boundary without modifying saved model bundles. Candidates with confidence at or above the threshold receive label 1 (`confirmed_slick`), while candidates below are labeled 0 (`lookalike_suppressed`).
* **AC-4**: Complete four deliverable export suite. For each evaluated scene, the pipeline creates the designated output directory and exports: (1) `scene_spills.geojson` containing confirmed slick polygons with comprehensive physical and geometric properties, (2) binary raster `mask.png` where confirmed slick pixels are marked 255 and ocean background is 0 matching scene dimensions, (3) `result.json` summarizing execution runtime, vessel counts, candidate statistics, and classification breakdown, and (4) `scene_candidates.geojson` containing all segmented candidate polygons tagged with their classification status and confidence.
* **AC-5**: Robust edge case and calm water handling. If a SAR scene contains zero candidate dark spots (e.g. calm sea clutter without low backscatter areas), the pipeline completes without crashing, exporting an empty `scene_spills.geojson` FeatureCollection, an all zero black `mask.png` matching input raster dimensions, and `result.json` recording 0 confirmed slicks and 0 candidates with exit code 0.

## Decision

**Chosen option**: Option 1: Explicit mode with smart default and sequential cascade in `testing/detect_oil.py`.

We enhance `testing/detect_oil.py` by integrating the four tested modules into a unified sequential cascade with a `--mode` selector, automatic model resolution, and standard deliverable exports.

## Rationale

Reasoning and options: see [rationale.md](rationale.md).

## Feature design

**Data model sketch**:

```python
@dataclass
class PipelineRunConfig:
    input_path: str                            # Path to input SAR GeoTIFF scene
    output_dir: str                            # Destination directory for deliverables
    mode: str                                  # Detection mode: "two_stage", "unet", or "rf_pixel"
    obia_model_path: str                       # Path to trained OBIA Random Forest joblib bundle
    unet_model_path: str                       # Path to U-Net PyTorch weights file
    pixel_rf_path: str                         # Path to legacy pixel Random Forest model
    threshold: Optional[float]                 # Operating probability threshold override
    enable_vessel_masking: bool                # Whether to compute vessel shadow corridors
    enable_candidate_export: bool              # Whether to export scene_candidates.geojson
    device: str                                # Inference compute device ("cpu" or "cuda")

@dataclass
class TwoStagePipelineResult:
    scene_id: str                              # Identifier extracted from input filename or directory
    pipeline_mode: str                         # Selected execution mode ("two_stage")
    total_vessels_detected: int                # Count of bright vessel reflectors detected
    shadow_pixels_masked: int                  # Count of ocean pixels masked in shadow corridors
    total_candidates_segmented: int            # Count of segmented candidate polygons
    total_candidate_area_km2: float            # Combined geodesic area of all candidates
    confirmed_slicks_count: int                # Count of confirmed mineral oil spill polygons
    suppressed_lookalikes_count: int           # Count of suppressed lookalike polygons
    confirmed_slicks_area_km2: float           # Combined geodesic area of confirmed slicks
    execution_time_seconds: float              # Total wall clock processing time
    deliverables: dict[str, str]               # Mapping of artifact names to file paths on disk
```

**Deliverables schema (`result.json`)**:

```json
{
  "scene_id": "det_20260917163850_004aae",
  "pipeline_mode": "two_stage",
  "model_path": "testing/best_obia_rf.joblib",
  "decision_threshold": 0.60,
  "execution_time_seconds": 12.45,
  "vessel_masking": {
    "vessels_detected": 4,
    "shadow_pixels_masked": 1250
  },
  "candidate_segmentation": {
    "total_candidates": 3576,
    "total_candidate_area_km2": 45.20
  },
  "object_classification": {
    "confirmed_slicks": 1,
    "suppressed_lookalikes": 3575,
    "confirmed_area_km2": 3.84
  },
  "deliverables": {
    "mask_png": "testing/det_20260917163850_004aae/mask.png",
    "spills_geojson": "testing/det_20260917163850_004aae/scene_spills.geojson",
    "candidates_geojson": "testing/det_20260917163850_004aae/scene_candidates.geojson",
    "result_json": "testing/det_20260917163850_004aae/result.json"
  }
}
```

**State transitions and execution flow**:

```
[SAR GeoTIFF Scene]
         |
         v
[Vessel Reflector & Shadow Masking] (testing/vessel_masking.py)
         |
         v
[Ocean Background Damping Segmentation] (testing/segmentation.py)
         |
         v
[17 Physical & Geometric Feature Extraction] (testing/feature_extractor.py)
         |
         v
[Object Level Random Forest Inference] (testing/obia_classifier.py)
         |
         +--> Probability >= Threshold --> [Confirmed Mineral Oil Slick]
         |                                          |
         +--> Probability <  Threshold --> [Suppressed Lookalike]
                                                    |
                                                    v
                                      [Deliverable File Generation]
                                      - scene_spills.geojson
                                      - mask.png
                                      - result.json
                                      - scene_candidates.geojson
```

**API surface**:

| Interface | Method or Signature | Key inputs | Key outputs | Auth | Key errors |
|---|---|---|---|---|---|
| CLI execution | `python testing/detect_oil.py [options]` | `--input`, `--output`, `--mode`, `--threshold`, `--model`, `--obia-model` | Deliverable files in output directory, exit code 0 | Local CLI | FileNotFoundError, ValueError on invalid raster |
| Programmatic pipeline runner | `run_two_stage_pipeline(config: PipelineRunConfig) -> TwoStagePipelineResult` | `config`: `PipelineRunConfig` | `TwoStagePipelineResult` dataclass instance | Programmatic Python | FileNotFoundError, ValueError on dimension mismatch |
| Raster mask generator | `rasterize_confirmed_slicks(slicks: list[CandidateDarkSpot], shape: tuple[int, int], transform) -> np.ndarray` | Slicks list, output height & width, raster transform | 2D uint8 NumPy array of values 0 and 255 | Programmatic Python | ValueError on invalid dimensions |
| Spills GeoJSON exporter | `export_spills_geojson(slicks: list[CandidateDarkSpot], classification_map: dict, output_path: str, transform, crs) -> str` | Slicks list, classification results, output path, transform, crs | GeoJSON file path string | Programmatic Python | IOError on write failure |

**Value sourcing**:

| Action | Value produced or displayed | Source |
|---|---|---|
| Pipeline setup | `scene_id` | Derived from input path: parent directory name if filename starts with `scene`, else file stem |
| Vessel masking | `vessels_detected`, `shadow_corridors` | Computed by `mask_vessel_shadows` in `testing/vessel_masking.py` |
| Candidate segmentation | `candidate_polygons`, `damping_ratio` | Computed by `segment_candidate_dark_spots` in `testing/segmentation.py` |
| Area aggregation | `total_candidate_area_km2`, `confirmed_area_km2` | Geodesic `area_km2` summed across segmented and confirmed candidates |
| Feature extraction | Canonical 17 feature array | Computed by `extract_candidate_features` in `testing/feature_extractor.py` |
| Object classification | `confidence`, `predicted_label`, `status` | Computed by `ObjectClassifier.predict_candidates` in `testing/obia_classifier.py` |
| Deliverable export | `mask.png` | Rasterized from confirmed slick polygons using rasterio, fast pathing to zeros if empty |
| Deliverable export | `scene_spills.geojson` | GeoJSON FeatureCollection formatted from confirmed candidate geometries |
| Deliverable export | `candidates_geojson` path | Path to `scene_candidates.geojson`, or `null` in `result.json` if `--no-candidates` is passed |
| Deliverable export | `result.json` | JSON dictionary compiled from pipeline stage metrics and file paths |

**Key invariants**:

* Running `testing/detect_oil.py` without arguments on reference scene `testing/det_20260917163850_004aae/scene.tif` must execute the two stage pipeline and correctly identify the reference slick `spot_001` with confidence above 0.95.
* Argument precedence: An explicit `--mode` flag always takes precedence. If `--mode` is omitted, `--model` paths ending in `.pt` select `unet`, paths containing an OBIA bundle dictionary or ending with `obia` select `two_stage`, and other `.joblib` paths select `rf_pixel`. If neither `--mode` nor `--model` is provided, the CLI auto detects `two_stage` if `testing/best_obia_rf.joblib` exists, or `unet` if `testing/best_unet.pt` exists, raising an explicit `FileNotFoundError` if neither is found.
* Threshold resolution: The operating threshold is unpacked from the `TrainedClassifierBundle` dictionary inside `testing/best_obia_rf.joblib` via `load_object_classifier` (default 0.60), falling back to 0.60 for bare models, and overridden by `--threshold` when specified.
* Fast path for empty slicks: `rasterize_confirmed_slicks` checks for an empty slicks collection and returns `np.zeros(shape, dtype=np.uint8)` directly without calling `rasterio.features.rasterize` to prevent geometry errors.
* Confirmed slick raster mask `mask.png` must match the exact pixel height and width of the input SAR raster.
* Empty scenes with zero dark spot candidates must produce valid deliverables without throwing unhandled exceptions.

**Security model**:

* Local execution environment without remote network endpoints.
* Model bundles and SAR rasters are read from the local filesystem with standard read permissions.

**Configuration required**:

No external environment variables or cloud credentials required. CLI options:
* `--input`: Path to input SAR GeoTIFF (default canonical reference scene).
* `--output`: Directory to store deliverables (default scene parent directory).
* `--mode`: Detection mode (`two_stage`, `unet`, `rf_pixel`).
* `--threshold`: Decision probability cutoff override (default uses bundle value 0.60).
* `--obia-model`: Path to object level Random Forest model bundle (default `testing/best_obia_rf.joblib`).
* `--model`: Generic model path supporting backward compatible usage.
* `--no-vessel-masking`: Flag to bypass vessel shadow masking.
* `--no-candidates`: Flag to skip intermediate candidate GeoJSON export.
* `--device`: Compute device for neural network inference (`cpu` or `cuda`).

**Critical test scenarios**:

* Happy path: Run two stage detection on reference scene `testing/det_20260917163850_004aae/scene.tif`, verifying that `spot_001` is confirmed as mineral oil spill with confidence > 0.95 and exported to `scene_spills.geojson` and `mask.png`, verifies **AC-1**, **AC-2**, **AC-4**.
* Decision threshold override: Run two stage detection with `--threshold 0.99`, verifying that borderline candidates are suppressed according to the higher cutoff, verifies **AC-3**.
* Zero candidate edge case: Run two stage detection on a synthetic SAR raster with high backscatter sea clutter containing no low backscatter spots, verifying graceful export of empty deliverables with exit code 0, verifies **AC-5**.
* Backward compatibility U-Net: Run detection specifying `--model testing/best_unet.pt`, verifying that U-Net inference executes properly and outputs `mask.png`, verifies **AC-1**.
* Backward compatibility pixel Random Forest: Run detection specifying `--model testing/best_model.joblib`, verifying that pixel baseline executes properly with vessel masking, verifies **AC-1**.

## Build plan

1. Implement `PipelineRunConfig`, `TwoStagePipelineResult`, and helper functions for rasterizing confirmed slicks and formatting `result.json` in `testing/detect_oil.py`, satisfies **AC-2**, **AC-4**.
2. Enhance `testing/detect_oil.py` argument parsing to add `--mode {two_stage, unet, rf_pixel}` and `--obia-model` while retaining backward compatibility for `--model`, satisfies **AC-1**, **AC-3**.
3. Wire the sequential cascade in `run_two_stage_pipeline` chaining vessel masking, candidate segmentation, feature extraction, and `ObjectClassifier` inference, satisfies **AC-2**, **AC-3**.
4. Implement deliverable generation logic writing `scene_spills.geojson`, binary `mask.png`, `result.json`, and `scene_candidates.geojson`, with zero candidate empty handling, satisfies **AC-4**, **AC-5**.
5. Create comprehensive automated test suite `testing/test_detect_oil.py` covering all CLI modes, deliverable structures, threshold overrides, and calm water edge cases, satisfies **AC-1**, **AC-2**, **AC-3**, **AC-4**, **AC-5**.

## Consequences

**Positive**:
* Unifies all four tested detection modules into a single, cohesive command line workflow.
* Delivers complete geospatial and raster deliverables required for downstream attribution and surveillance dashboards.
* Automatically selects the high precision two stage OBIA pipeline without breaking legacy U-Net or pixel baseline workflows.

**Negative / tradeoffs**:
* Running the full two stage cascade (vessel masking, background damping convolution, feature extraction, and tree voting) takes approximately 10 to 15 seconds per 512x512 SAR scene on CPU, compared to 1 to 2 seconds for pixel models.

**Neutral**:
* Generated deliverables match standard naming conventions (`mask.png`, `scene_spills.geojson`, `result.json`), allowing drop in compatibility with existing visualization tools.

## Follow-up

* [ ] Connect operational surveillance dashboard `scripts/dashboard/app.py` to ingest `scene_spills.geojson` and `result.json` generated by the two stage pipeline.

## Migration plan

**Strategy**: Strangler pattern with backward compatible CLI arguments.
**Phases**:
1. Add the two stage pipeline implementation and `--mode` argument alongside existing U-Net and pixel Random Forest functions in `testing/detect_oil.py`.
2. Set default mode resolution to select `two_stage` when `best_obia_rf.joblib` is present while keeping `--model` commands functional.
3. Validate operational consistency across all test suites and existing scripts before considering deprecation of legacy pixel models.
**Rollback**: Revert `testing/detect_oil.py` to previous revision if unforeseen regressions occur.
**Risks**: Minor risk of flag confusion if users supply conflicting `--model` and `--mode` parameters, mitigated by clear precedence and validation checks in argument parsing.
