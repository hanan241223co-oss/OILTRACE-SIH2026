# 0004. Object Level Random Forest Slick Classifier

**Date**: 2026-09-19
**Status**: Accepted

## Summary

This specification establishes an object level Random Forest classifier for marine oil spill detection. It trains on 17 dimensional physical and geometric feature vectors extracted from candidate radar dark spot polygons across multiple satellite scenes. The classifier separates genuine mineral oil slicks from natural ocean lookalikes using calibrated probabilities, grouped scene cross validation, and a high precision decision threshold.

## Requirements

**User stories**:
* As a marine surveillance operator, I want candidate dark spots classified into verified mineral oil spills or suppressed lookalikes with confidence probabilities, so that false alarms from natural ocean phenomena are minimized.
* As an automated pipeline developer, I want a trained model bundle and inference interface that consumes canonical 17 feature vectors and outputs structured classification decisions, so that detection results integrate into surveillance reports and vector exports.

**Acceptance criteria**:

* **AC-1**: Multi scene training dataset ingestion and label assignment. The training pipeline ingests SAR scenes and optional raster ground truth masks via a structured JSON manifest (`testing/train_manifest.json` formatted as a top level list of scene objects), verifies that mask raster dimensions match scene dimensions exactly, runs candidate dark spot segmentation from Feature 9 and canonical 17 feature extraction from Feature 10, and assigns binary ground truth labels to each candidate polygon using a verified spatial matching rule (positive label 1 if pixel intersection with ground truth mask is at least 40 pixels and at least 20 percent polygon area overlap, else label 0). Scenes with null or missing masks treat all candidate polygons as confirmed negative lookalikes with label 0.
* **AC-2**: Random Forest classifier training with class balance. The system trains a scikit-learn `RandomForestClassifier` on the assembled multi scene candidate feature matrix using 150 estimator trees, maximum tree depth of 12, minimum samples split of 5, `random_state=42`, and `class_weight='balanced'` to compensate for lookalike majority class imbalance. Training evaluates model generalization using grouped five fold cross validation (`StratifiedGroupKFold` grouped by `scene_id` to prevent data leakage across candidates from the same scene, with graceful fallback to `StratifiedKFold` when positive scene count is under 5), reporting precision, recall, F1 score, ROC-AUC, and flattened confusion matrix counts (`cm_tn`, `cm_fp`, `cm_fn`, `cm_tp`).
* **AC-3**: High precision decision thresholding and confidence calibration. The classifier outputs continuous confidence probabilities derived from ensemble tree voting via `predict_proba`. A configurable decision threshold (default 0.60) prioritizes operational precision by suppressing borderline natural lookalikes, producing a discrete binary label (1 for confirmed slick, 0 for suppressed lookalike) and a descriptive classification status string (`confirmed_slick` or `lookalike_suppressed`). Inference methods accept an optional `decision_threshold` parameter allowing runtime operational overrides.
* **AC-4**: Serialized model bundle packaging and validation. The training routine packages the trained classifier, ordered canonical feature names list, decision threshold, training timestamp, training dataset summary (standard keys `total_scenes`, `total_samples`, `positive_samples`, `negative_samples`), feature median array, and evaluation metrics into a validated dictionary bundle saved via joblib (`testing/best_obia_rf.joblib`). Deserialization verifies bundle keys, model instance class, and exact alignment with the 17 canonical feature names, raising explicit errors on corrupted or incompatible files.
* **AC-5**: Programmatic inference API and missing value imputation. The module `testing/obia_classifier.py` provides an `ObjectClassifier` class supporting both high level `predict_candidates(candidates: list[CandidateDarkSpot], decision_threshold: float | None = None) -> list[ObjectClassificationResult]` and low level `predict_features(feature_matrix: np.ndarray, decision_threshold: float | None = None) -> tuple[np.ndarray, np.ndarray]`. The inference pipeline gracefully handles empty candidate collections by returning empty result lists, checks for feature schema alignment, converts any infinite values to NaN, and imputes NaNs using training median statistics (with a fallback value of 0.0 for all NaN columns) before forest evaluation.

## Decision

**Chosen option**: Option 1: Scikit-learn Random Forest with Balanced Class Weighting, Grouped Cross Validation, and Canonical 17 Features.

We train an ensemble of 150 decision trees on the standardized 17 dimensional object feature vector, using balanced class weighting to handle lookalike imbalance, grouped cross validation by scene to avoid data leakage, robust median imputation, and an operating threshold of 0.60 to deliver high precision slick detection.

## Rationale

Reasoning and options: see [rationale.md](rationale.md).

## Feature design

**Data model sketch**:

```python
@dataclass
class SceneManifestEntry:
    scene_id: str                              # Unique scene identifier (e.g. det_20260917163850_004aae)
    scene_path: str                            # Relative or absolute path to SAR GeoTIFF scene
    mask_path: str | None                      # Optional path to ground truth raster mask (None for lookalikes)
    scene_tag: str                             # Category tag (e.g. oil_scene, lookalike_scene)

@dataclass
class TrainingSampleRecord:
    sample_id: str                             # Unique sample identifier (e.g. det_004aae_spot_001)
    scene_id: str                              # Foreign reference linking to SceneManifestEntry
    candidate_id: str                          # Original candidate identifier from segmentation
    feature_vector: np.ndarray                 # 1D float32 array of shape (17,) matching CANONICAL_FEATURE_NAMES
    ground_truth_overlap_pixels: int           # Count of overlapping positive pixels with ground truth mask
    ground_truth_overlap_fraction: float       # Ratio of overlap pixels to candidate polygon pixel area
    label: int                                 # Ground truth binary class: 1 for oil spill, 0 for lookalike

@dataclass
class ObjectClassificationResult:
    candidate_id: str                          # Candidate polygon identifier matching CandidateDarkSpot
    predicted_label: int                       # Binary decision: 1 for confirmed oil spill, 0 for lookalike
    confidence: float                          # Probability score of oil spill class in range [0.0, 1.0]
    decision_threshold: float                  # Operating probability threshold applied (e.g. 0.60)
    classification_status: str                 # "confirmed_slick" if predicted_label == 1 else "lookalike_suppressed"

@dataclass
class TrainedClassifierBundle:
    model: RandomForestClassifier              # Fitted scikit-learn random forest estimator
    feature_names: list[str]                   # Ordered list of 17 canonical feature names
    decision_threshold: float                  # Operating threshold for positive classification
    created_at: str                            # ISO 8601 creation timestamp string
    metrics: dict[str, float]                  # Validation metrics: precision, recall, f1_score, roc_auc, cm_tn, cm_fp, cm_fn, cm_tp
    training_manifest_summary: dict[str, int]  # Summary dictionary: total_scenes, total_samples, positive_samples, negative_samples
    feature_medians: np.ndarray                # 1D float32 array of shape (17,) for robust missing value imputation
```

**State transitions**:

```
[Candidate Dark Spot]
         |
         v
[Feature Extraction (17 canonical features)]
         |
         v
[Imputation (inf -> NaN, NaN -> training medians)]
         |
         v
[Random Forest Evaluation (ensemble voting)]
         |
         +--> Probability >= Operating Threshold --> [Confirmed Mineral Oil Slick]
         |
         +--> Probability <  Operating Threshold --> [Suppressed Natural Lookalike]
```

**API surface**:

| Interface | Method or Signature | Key inputs | Key outputs | Auth | Key errors |
|---|---|---|---|---|---|
| CLI training script | `python testing/train_obia_rf.py` | `--manifest` (path), `--output` (path), `--threshold` (float), `--cv` (int) | Trained joblib bundle, stdout metrics report | Local CLI | ValueError on empty manifest or single class dataset, FileNotFoundError |
| Model loader | `load_object_classifier(model_path: str) -> ObjectClassifier` | `model_path`: str | `ObjectClassifier` instance | Local filesystem | FileNotFoundError, ValueError on invalid bundle keys or feature mismatch |
| High level candidate prediction | `ObjectClassifier.predict_candidates(candidates: list[CandidateDarkSpot], decision_threshold: float \| None = None) -> list[ObjectClassificationResult]` | `candidates`: list of `CandidateDarkSpot` objects, optional threshold float | list of `ObjectClassificationResult` | Programmatic Python | ValueError on missing feature vector or dimension mismatch |
| Low level feature matrix prediction | `ObjectClassifier.predict_features(feature_matrix: np.ndarray, decision_threshold: float \| None = None) -> tuple[np.ndarray, np.ndarray]` | `feature_matrix`: 2D float array of shape (N, 17), optional threshold float | `(labels, probabilities)`: two 1D NumPy arrays of shape (N,) | Programmatic Python | ValueError on invalid 2D shape or incorrect column count != 17 |

**Value sourcing**:

| Action | Value produced or displayed | Source |
|---|---|---|
| Dataset ingestion | `ground_truth_overlap_pixels` | Raster intersection between candidate polygon mask and `mask_path` GeoTIFF |
| Dataset ingestion | `ground_truth_overlap_fraction` | `ground_truth_overlap_pixels` divided by candidate pixel area |
| Dataset ingestion | `label` | Derived via AC-1 rule: 1 if `overlap_pixels >= 40` and `overlap_fraction >= 0.20`, else 0 |
| Model training | `metrics` | Computed from grouped cross validation predictions against ground truth labels |
| Model training | `feature_medians` | Column medians of training feature matrix computed after converting infinities to NaNs, with 0.0 fallback |
| Model training | `training_manifest_summary` | Aggregate counts of scenes, total candidates, positive samples, and negative samples |
| Inference prediction | `confidence` | Extracted from `model.predict_proba(X)[:, 1]` |
| Inference prediction | `predicted_label` | Derived via AC-3 rule: 1 if `confidence >= threshold` else 0 |
| Inference prediction | `classification_status` | String mapping: `confirmed_slick` if `predicted_label == 1` else `lookalike_suppressed` |
| Imputation | Imputed feature values | Replaces any infinite values with NaNs, then fills NaNs with corresponding `feature_medians` |

**Key invariants**:

* The feature vector must always follow the exact 17 column ordering defined in `CANONICAL_FEATURE_NAMES` from Feature 10.
* Input feature matrices with infinite or missing values must be imputed with training medians before tree evaluation; the classifier never crashes on NaNs or infinities.
* Empty candidate lists must return empty classification result lists immediately without triggering array reshaping errors.
* Binary classification requires both classes; the training routine halts with an explicit error if all samples belong to only one class.
* Ground truth mask rasters must match scene raster dimensions exactly; dimension mismatches raise a ValueError.

**Security model**:

* Local execution environment without remote network endpoints.
* Model bundles are stored in local filesystem with standard read and write permissions.
* The unpickling routine verifies bundle dictionary keys and checks that the deserialized model matches expected scikit-learn estimator types before use.

**Configuration required**:

No external environment variables or cloud credentials required. The training script accepts optional CLI arguments:
* `--manifest`: Path to training manifest JSON (default `testing/train_manifest.json`).
* `--output`: Destination path for saved joblib bundle (default `testing/best_obia_rf.joblib`).
* `--threshold`: Operating probability cutoff (default 0.60).
* `--cv`: Number of cross validation folds (default 5).

**Critical test scenarios**:

* Happy path: Ingest multi scene manifest, extract 17 features, fit Random Forest, and produce validated joblib bundle with precision above 0.85, verifies AC-1, AC-2, AC-4.
* High precision thresholding: Verify that candidate polygons with probabilities between 0.50 and 0.59 are labeled as 0 with status `lookalike_suppressed`, while candidates with probability 0.60 and above are labeled 1 with status `confirmed_slick`, verifies AC-3.
* Threshold override: Passing `decision_threshold=0.45` to `predict_candidates` reclassifies candidates using the override threshold without mutating the underlying bundle, verifies AC-3.
* Empty candidates handling: Calling `predict_candidates([])` returns an empty list without raising exceptions, verifies AC-5.
* Missing value and infinite value imputation: Providing candidate feature vectors with injected NaN and infinite values produces valid predictions matching median imputed inputs, verifies AC-5.
* Grouped cross validation leakage prevention: Confirm that candidate polygons from the same `scene_id` reside strictly within the same validation fold, verifies AC-2.
* Integrity check: Attempting to load an invalid bundle dictionary or one with mismatching feature names raises an explicit ValueError, verifies AC-4.

## Build plan

1. [x] Create dataset manifest schema, default manifest `testing/train_manifest.json`, and multi scene candidate sample builder with ground truth spatial matching logic in `testing/train_obia_rf.py`, satisfies **AC-1**.
2. [x] Implement Random Forest model training, balanced class weighting, robust median feature imputation, grouped five fold cross validation, and performance reporting in `testing/train_obia_rf.py`, satisfies **AC-2**.
3. [x] Implement model bundle serialization, metadata packaging, and bundle integrity validation functions, satisfies **AC-4**.
4. [x] Create dedicated inference module `testing/obia_classifier.py` containing `ObjectClassifier`, runtime threshold override support, and robust median missing value imputation, satisfies **AC-3**, **AC-5**.
5. [x] Implement comprehensive unit and integration test suite `testing/test_obia_classifier.py` covering all acceptance criteria, verifies **AC-1**, **AC-2**, **AC-3**, **AC-4**, **AC-5**.

## Consequences

**Positive**:

* Eliminates false alarms caused by natural ocean lookalikes, biogenic films, and vessel radar shadows that confound pixel level models.
* Grouped cross validation ensures honest performance estimates without data leakage between candidates in the same scene.
* Standardized 17 dimensional feature vector provides transparent and interpretable classification behavior.
* Clear separation between training pipeline (`testing/train_obia_rf.py`) and runtime inference engine (`testing/obia_classifier.py`).

**Negative / tradeoffs**:

* Training requires multi scene GeoTIFF rasters and ground truth masks to build a balanced dataset.
* Setting a high precision threshold (0.60) suppresses false alarms but may classify faint or heavily weathered mineral slicks as lookalikes.

**Neutral**:

* Output model bundle file size is compact (under 2 megabytes), making it lightweight to store and distribute.
* Fully backward compatible with existing candidate segmentation and feature extraction modules.

## Follow-up

* [ ] Integrate `ObjectClassifier` into operational surveillance script `testing/detect_oil.py` as part of Feature 12.
