# Verify: Object Level Random Forest Slick Classifier · spec 0004 · updated 2026-09-19
_Steps derived from spec 0004 acceptance criteria. `/check verify` runs these; `/test` locks the durable ones._

## Commands
* [x] `python3 -c "import json, os; m=json.load(open('testing/train_manifest.json')); assert isinstance(m, list) and len(m) >= 1; assert 'scene_path' in m[0] and 'scene_id' in m[0]"` -> Verify training manifest schema is a list of scene objects -> AC-1
* [x] `python3 -c "from testing.train_obia_rf import compute_ground_truth_label; assert compute_ground_truth_label(50, 200, 0.25) == 1; assert compute_ground_truth_label(30, 200, 0.15) == 0; assert compute_ground_truth_label(100, 1000, 0.10) == 0"` -> Spatial ground truth matching rule correctly assigns labels based on pixel count and area fraction -> AC-1
* [x] `python3 -c "from testing.train_obia_rf import train_model_from_manifest; bundle = train_model_from_manifest('testing/train_manifest.json'); assert bundle.model is not None and len(bundle.feature_names) == 17; assert 'f1_score' in bundle.metrics and 'cm_tp' in bundle.metrics"` -> Model training pipeline trains random forest on multi scene manifest and computes grouped validation metrics -> AC-1, AC-2, AC-4
* [x] `python3 -c "import joblib; b = joblib.load('testing/best_obia_rf.joblib'); assert len(b['feature_names']) == 17 and b['decision_threshold'] == 0.60; assert 'cm_tn' in b['metrics'] and 'feature_medians' in b"` -> Serialized joblib bundle contains verified model instance, 17 feature names, operating threshold, metrics, and feature medians -> AC-4
* [x] `python3 -c "from testing.obia_classifier import load_object_classifier; clf = load_object_classifier('testing/best_obia_rf.joblib'); assert clf.decision_threshold == 0.60; assert clf.feature_names[0] == 'area_km2'"` -> Model loader deserializes and validates bundle integrity -> AC-4, AC-5
* [x] `python3 -c "from testing.obia_classifier import load_object_classifier; import numpy as np; clf = load_object_classifier('testing/best_obia_rf.joblib'); labels, probs = clf.predict_features(np.zeros((3, 17), dtype=np.float32)); assert len(labels) == 3 and len(probs) == 3"` -> Low level matrix inference outputs binary labels and probability scores -> AC-3, AC-5
* [x] `python3 -c "from testing.obia_classifier import load_object_classifier; import numpy as np; clf = load_object_classifier('testing/best_obia_rf.joblib'); x = np.zeros((1, 17), dtype=np.float32); x[0, 5] = np.inf; x[0, 2] = np.nan; labels, probs = clf.predict_features(x); assert not np.isnan(probs[0])"` -> Missing value and infinite value imputation cleanly handles inputs without crashing -> AC-5
* [x] `python3 -c "from testing.obia_classifier import load_object_classifier; import numpy as np; clf = load_object_classifier('testing/best_obia_rf.joblib'); x = np.zeros((1, 17), dtype=np.float32); l1, p1 = clf.predict_features(x, decision_threshold=0.01); l2, p2 = clf.predict_features(x, decision_threshold=0.99); assert l1[0] >= l2[0]"` -> Threshold override parameter dynamically adjusts positive classification cutoff -> AC-3
* [x] `python3 -c "from testing.obia_classifier import load_object_classifier; clf = load_object_classifier('testing/best_obia_rf.joblib'); res = clf.predict_candidates([]); assert res == []"` -> Empty candidate collection returns empty list gracefully -> AC-5
* [x] `python3 -c "from testing.obia_classifier import ObjectClassificationResult; r = ObjectClassificationResult('c1', 1, 0.75, 0.60, 'confirmed_slick'); assert r.classification_status == 'confirmed_slick'"` -> Classification result entity formats status string correctly -> AC-3
* [x] `pytest -q testing/test_obia_classifier.py` -> Automated test suite passes all unit and integration tests -> AC-1, AC-2, AC-3, AC-4, AC-5

## Acceptance-criteria coverage
* AC-1 (Multi scene dataset ingestion and ground truth matching): covered by commands 1, 2, 3, 11
* AC-2 (Random Forest training with class balance and grouped cross validation): covered by commands 3, 11
* AC-3 (High precision thresholding and confidence calibration): covered by commands 6, 8, 10, 11
* AC-4 (Serialized model bundle packaging and validation): covered by commands 3, 4, 5, 11
* AC-5 (Programmatic inference API and missing value imputation): covered by commands 5, 6, 7, 9, 11
