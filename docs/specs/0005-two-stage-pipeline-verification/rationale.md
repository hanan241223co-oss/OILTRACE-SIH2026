# 0005. Two Stage Pipeline Verification in Testing · Rationale

## Context

The OILTRACE project aims to detect marine oil spills accurately in satellite radar imagery while minimizing false alarms from natural ocean phenomena and ship radar shadows. Across Slices 8 through 11, the team developed and verified four distinct modular components in the testing environment:

1. Autonomous vessel masking (`testing/vessel_masking.py`, Spec 0001) for detecting bright reflectors and projecting directional radar shadow corridors.
2. Adaptive candidate segmentation (`testing/segmentation.py`, Spec 0002) for segmenting dark spot regions using ocean background damping contrast.
3. Physical and geometric feature extraction (`testing/feature_extractor.py`, Spec 0003) for calculating 17 dimensional morphology, radar damping, and boundary sharpness features.
4. Object level Random Forest classifier (`testing/obia_classifier.py` and `testing/train_obia_rf.py`, Spec 0004) for distinguishing confirmed mineral oil slicks from suppressed natural lookalikes.

Currently, `testing/detect_oil.py` serves as the primary inference script, supporting deep learning U-Net inference and a legacy pixel level Random Forest baseline. While candidate segmentation and feature extraction flags were partially wired into `detect_oil.py` during earlier slices, the script lacks an end to end pipeline that automatically chains vessel masking, candidate segmentation, feature extraction, and the trained object classifier into a single executable workflow.

A unified operational entry point is needed in `testing/detect_oil.py` that executes this complete sequence, outputs verified geospatial deliverables, and maintains seamless backward compatibility with existing test scripts and pipelines.

## Options considered

### Option 1: Explicit mode with smart default and sequential cascade in detect_oil.py

Enhance the existing `testing/detect_oil.py` script by introducing an explicit `--mode {two_stage, unet, rf_pixel}` CLI argument. When `--mode` is not explicitly provided, the CLI inspects available model weights and defaults to `two_stage` when `testing/best_obia_rf.joblib` is present. The pipeline chains the four modular stages in a sequential cascade and outputs standard deliverables: `scene_spills.geojson`, `mask.png`, `result.json`, and `scene_candidates.geojson`.

**Pros**:
* Unifies all detection capabilities within the primary CLI tool familiar to operators and developers.
* Follows the strangler pattern, keeping legacy U-Net and pixel Random Forest modes fully functional.
* Clear sequential data flow makes debugging and intermediate asset inspection straightforward.
* Produces consistent deliverables matching downstream ingestion requirements.

**Cons**:
* Increases the code size and import responsibilities of `testing/detect_oil.py`.

### Option 2: Post classification spatial filtering

Run candidate segmentation and Random Forest classification on all dark spots first, without pre masking vessel shadows. Then, perform a spatial intersection query between classified slicks and vessel shadow corridors to filter out false positives retroactively.

**Pros**:
* Allows the classifier to see every dark spot without prior spatial modification.

**Cons**:
* Wastes compute resources extracting 17 features and evaluating trees on hundreds of shadow candidates that could be eliminated early.
* Increases the risk of false positives if a shadow candidate receives high classification confidence due to boundary contrast.
* Breaks the natural pipeline flow established in earlier specifications.

### Option 3: Separate standalone OBIA script

Create a separate script, such as `testing/detect_oil_obia.py`, dedicated exclusively to the two stage object based pipeline, leaving `testing/detect_oil.py` untouched.

**Pros**:
* Completely isolates new code from existing legacy scripts, eliminating any regression risk in `testing/detect_oil.py`.

**Cons**:
* Creates duplicate CLI boilerplates, argument parsers, and raster loading routines.
* Forces surveillance operators and test suites to maintain multiple execution commands for different models.
* Contradicts the goal of having a single operational detection tool in testing.

## Rationale

Option 1 is selected because it delivers a unified, coherent surveillance CLI while honoring operational reality and avoiding code sprawl.

By enhancing `testing/detect_oil.py` with an explicit `--mode` selector and smart defaults, existing automated tests and documentation commands continue to work without disruption. Operators can easily compare the deep learning U-Net model against the two stage object based Random Forest pipeline using the same CLI interface. Furthermore, the sequential cascade ensures that vessel radar shadows are suppressed early, minimizing downstream computation and preventing lookalike false alarms.
