# Scope: OILTRACE Marine Oil Spill Detection & Attribution

Automated satellite radar surveillance platform for detecting marine oil spills, suppressing ocean lookalikes, and attributing spills to candidate vessels.

**Build approach:** Tracer Bullet (prove the entire two stage object classification thread on real data first).
**Workflow:** Beta (check verify on real imagery, then test with regression checks). The project default level of rigor. /architect is the recommended first stop for a feature with a real decision, but skippable when you already know the build. Any feature can carry its own tag to do more or less.

_These are recommendations to keep your build orderly, not requirements. Skip anything that does not fit: if you already know how to build a feature, use /develop and skip /architect. You decide when a feature is done._

## At a glance

| # | Feature | Phase | Status |
|---|---------|-------|--------|
| 1 | SAR Calibration & Speckle Filtering | Foundation | existing |
| 2 | Polarimetric Pixel Feature Engineering | Foundation | existing |
| 3 | Pixel Level Random Forest Baseline | Foundation | existing |
| 4 | Spatial Geometry & GeoJSON Polygonization | Foundation | existing |
| 5 | Hydrodynamic Drift Trajectory Modeling | Foundation | existing |
| 6 | AIS Candidate Vessel Attribution | Foundation | existing |
| 7 | Operational Surveillance Dashboard | Foundation | existing |
| 8 | Autonomous Vessel & Bright Target Masking | Slice 1 | done |
| 9 | Adaptive Dark Spot Candidate Segmentation | Slice 1 | done |
| 10 | Physical & Geometric Object Feature Extractor | Slice 1 | done |
| 11 | Object Level Random Forest Slick Classifier | Slice 1 | done |
| 12 | Two Stage Pipeline Verification in Testing | Slice 1 | done |

## Existing features (enrolled brownfield context)

### 1. SAR Calibration & Speckle Filtering · existing
Pre workflow calibration: radiometric scaling to decibels, outlier clipping, and adaptive spatial Lee filtering for speckle noise reduction. code in `scripts/preprocessing/preprocess_sar.py`

### 2. Polarimetric Pixel Feature Engineering · existing
Four band normalized pixel feature stack: normalized VV backscatter, normalized VH backscatter, polarimetric difference ratio, and local standard deviation texture. code in `scripts/preprocessing/feature_engineering.py`

### 3. Pixel Level Random Forest Baseline · existing
Initial scikit-learn random forest classifier trained on four scalar pixel channels, running fast per pixel probability estimation. code in `scripts/training/inference.py` and `testing/best_model.joblib`

### 4. Spatial Geometry & GeoJSON Polygonization · existing
Connected component labeling, image coordinate to geographic coordinate mapping, bounding box calculation, and GeoJSON vector polygon export. code in `scripts/training/polygonise.py`

### 5. Hydrodynamic Drift Trajectory Modeling · existing
Lagrangian particle tracking simulating backward hindcast trajectory to locate spill origins and forward forecast dispersion over time. code in `scripts/attribution/drift_trajectory.py`

### 6. AIS Candidate Vessel Attribution · existing
Spatiotemporal intersection between simulated backward drift trajectories and historical AIS ship tracks, ranking suspect vessels by proximity and evidence score. code in `scripts/attribution/vessel_attribution.py`

### 7. Operational Surveillance Dashboard · existing
Streamlit web application with interactive Folium geospatial maps, detection inspection, drift trajectory review, and candidate vessel evidence dossiers. code in `scripts/dashboard/app.py`

## Slice 1: Two Stage Object Based Oil Detection

### 8. Autonomous Vessel & Bright Target Masking · done
Detect strong radar corner reflections from metal ships and offshore platforms, calculate directional radar shadow corridors behind them, and eliminate false alarm dark patches caused by vessels.
**Done when:** bright metallic reflectors are identified, shadow geometry corridors are calculated, and false positives adjacent to vessels are suppressed without clipping true slicks.
- [x] Design it (spec): [docs/specs/0001-autonomous-vessel-masking/index.md](../specs/0001-autonomous-vessel-masking/index.md)
- [x] Build it: `/develop autonomous vessel & bright target masking`
  - [x] Vectorized dual polarization vessel reflector detection
  - [x] Directional shadow corridor projection and radial fallback
  - [x] Physical damping contrast check and slick preservation override
  - [x] GeoJSON point export and integration with detect_oil.py
  code in `testing/vessel_masking.py` and `testing/detect_oil.py`
- [x] Verify it: `/check verify autonomous vessel & bright target masking`
- [x] Test it: `/test`

### 9. Adaptive Dark Spot Candidate Segmentation · done
Segment candidate low backscatter ocean regions using local adaptive thresholding and morphological closing, extracting coherent candidate polygons while rejecting single pixel speckle.
**Done when:** candidate dark spot regions are segmented into connected polygons across varying sea clutter backgrounds without fragmenting continuous oil slicks.
- [x] Design it (spec): [docs/specs/0002-adaptive-candidate-segmentation/index.md](../specs/0002-adaptive-candidate-segmentation/index.md)
- [x] Build it: `/develop adaptive dark spot candidate segmentation`
  - [x] Moving window background clutter convolution and damping contrast calculation (AC-1, AC-2)
  - [x] Morphological closing, 8-connectivity labeling, and area filtering (AC-3)
  - [x] Geographic vector polygonization and geodesic area calculation (AC-5)
  - [x] Vessel masking integration and suspect vessel proximity linkage (AC-4)
  - [x] Pipeline integration in detect_oil.py with CLI flags and GeoJSON export (AC-4, AC-5)
  code in `testing/segmentation.py` and `testing/detect_oil.py`
- [x] Verify it: `/check verify adaptive dark spot candidate segmentation`
- [x] Test it: `/test`

### 10. Physical & Geometric Object Feature Extractor · done
Compute multi dimensional object features for each candidate polygon: elongation aspect ratio, boundary gradient sharpness, fractal border complexity, area, perimeter, and the physical radar damping ratio comparing slick backscatter against local surrounding ocean.
**Done when:** every candidate polygon receives a structured dictionary of physical and geometric features, including damping ratio and boundary sharpness.
- [x] Design it (spec): [docs/specs/0003-physical-geometric-feature-extractor/index.md](../specs/0003-physical-geometric-feature-extractor/index.md)
- [x] Build it: `/develop physical & geometric object feature extractor`
  - [x] Geometric morphology descriptors: elongation, circularity, complexity, fractal dimension, extent (AC-1)
  - [x] Physical radar backscatter statistics and texture standard deviation extraction (AC-2)
  - [x] Precomputed scene Sobel gradient and masked boundary sharpness calculation (AC-3)
  - [x] Vessel proximity context integration and numerical imputation (AC-4)
  - [x] Dual container packaging, DataFrame tabular export, and detect_oil.py CLI integration (AC-5)
  code in `testing/feature_extractor.py` and `testing/detect_oil.py`
- [x] Verify it: `/check verify physical & geometric object feature extractor`
- [x] Test it: `/test`

### 11. Object Level Random Forest Slick Classifier · done
Train a scikit-learn random forest model on object level feature vectors extracted from multi scene training data, classifying candidate polygons as confirmed mineral oil spill or natural lookalike with calibrated probabilities.
**Done when:** the classifier trains on polygon samples from the dataset manifest, achieves higher precision than the pixel baseline, and outputs clean binary decisions and confidence scores.
- [x] Design it (spec): [docs/specs/0004-object-level-random-forest-classifier/index.md](../specs/0004-object-level-random-forest-classifier/index.md)
- [x] Build it: `/develop object level random forest slick classifier`
  - [x] Multi scene training manifest ingestion and spatial ground truth matching (AC-1)
  - [x] Balanced Random Forest training, median imputation, and grouped cross validation (AC-2)
  - [x] Model bundle serialization and integrity verification (AC-4)
  - [x] Programmatic inference engine with threshold override and missing value imputation (AC-3, AC-5)
  code in `testing/train_obia_rf.py` and `testing/obia_classifier.py`
- [x] Verify it: `/check verify object level random forest slick classifier`
- [x] Test it: `/test`

### 12. Two Stage Pipeline Verification in Testing · done
Integrate candidate segmentation, object feature extraction, and the trained object classifier into `testing/detect_oil.py`, evaluating performance on real test scenes and exporting verified GeoJSON polygons and PNG masks.
**Done when:** running the updated testing pipeline on test scenes correctly identifies true oil slicks, eliminates vessel shadow false alarms, and exports clean deliverables.
- [x] Design it (spec): [docs/specs/0005-two-stage-pipeline-verification/index.md](../specs/0005-two-stage-pipeline-verification/index.md)
- [x] Build it: `/develop two stage pipeline verification in testing`
  - [x] Configuration and pipeline data models (AC-2, AC-4)
  - [x] CLI mode selection and argument precedence (AC-1, AC-3)
  - [x] End to end sequential detection cascade (AC-2, AC-3)
  - [x] Deliverables export and empty scene handling (AC-4, AC-5)
  code in `testing/detect_oil.py`
- [x] Verify it: `/check verify two stage pipeline verification in testing`
- [x] Test it: `/test`

## Deferred

Out of scope for the current build pass, kept so the plan stays honest.
- **AIS Live Stream Fusion**: dynamic vessel shadow validation using real time stream feeds · needs a decision
- **Numerical Weather Wind Field Integration**: automated ERA5 wind speed retrieval to calibrate physical damping ratio thresholds dynamically · needs a decision
- **Interactive Object Feature Inspection in Dashboard**: polygon level attribute breakdown and radar damping profile plots in Streamlit · needs a decision

## Legend

**The decision box.** Every feature carries exactly one, the sub-task whose label ends with `(spec)`. Its wording varies (`Design it (spec)` normally), so skills locate it by that `(spec)` suffix, never by an exact label. Every other box is an execution box and `/architect` never ticks one.

**Feature lifecycle**: the scope updates as a feature moves; each row is what it shows and who sets it:

| State | Set by | The feature shows |
|---|---|---|
| `planned` · needs a decision | `/scope` | one box: `Design it (spec): /architect <feature>` |
| `in-progress` (designed) | **`/architect` at spec capture** | `Design it` ticked; spec linked; `Build it: /develop <feature>` + **2 to 5 milestones**; the tier's closing boxes (`Verify it` Alpha+, `Test it` Beta+, `Review it` + `Document it` GA); any surfaced follow up enrolled |
| `in-progress` (building) | `/develop` | milestone sub-boxes tick one by one; code pointer filled |
| `in-progress` (verified) | `/check verify` | `Build it` + milestones ticked; `Verify it` ticked |
| `done` | **you, when you decide it is** (any skill sets it when you say so); `/sync` reconciles | boxes you ran ticked, skipped ones marked skipped; the tier's last stage (`Prototype` → after `/develop`; `Alpha` → after `/check verify`; `Beta`/`GA` → after `/test`) is the suggested point to call it done; `/sync` captures conventions |

- **Next step** = the first unticked box (always a command or a tracked milestone).
- **needs a decision** = run `/architect` first; otherwise straight to `/develop`. The tag drops once the spec is captured.
- **Atomic build tasks live in the spec's `## Build plan`, not here**: the scope carries only the milestone rollup.
- **Status** `planned` → `in-progress` → `done`, plus `existing` (pre workflow) and `dropped` (de-scoped, kept for history).
- **Workflow** (header line) is the project default, what runs after `/develop`: **Prototype** = nothing; **Alpha** = `/check verify`; **Beta** = `/check verify` then `/test`; **GA** = adds `/check review` then `/document`.
