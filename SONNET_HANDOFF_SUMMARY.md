# OILTRACE Handover & Operational Summary
Generated: 2026-09-17 20:38:00
Location: D:\OILTRACE\SONNET_HANDOFF_SUMMARY.md

======================================================================
CURRENT SYSTEM & PIPELINE STATUS
======================================================================
CURRENT DATASET: 11 verified valid scenes (9 oil_spill, 2 look_alike), 1 corrupt (00011.tif), 2,570 ground-truth masks.
DATASET DOWNLOAD STATUS: Upstream Zenodo servers timing out (HTTP 504 Gateway Time-out / Cloudflare 403). Download script D:\OILTRACE\scripts\download_dataset_subset.py ready to resume.
QGIS STATUS: Operational. Project D:\OILTRACE\qgis\OILTRACE.qgz verified with 7 layer groups (RAW SAR, Preprocessed, Training Data, Detection Results, Drift, AIS, Final Output).
SAR PREPROCESSING STATUS: Fully implemented in D:\OILTRACE\scripts\preprocessing\preprocess_sar.py (radiometric outlier clipping [-50 dB, +5 dB], Lee speckle filter, min-max normalisation).
FEATURE ENGINEERING STATUS: Operational in D:\OILTRACE\scripts\preprocessing\feature_engineering.py (4-band stack: VV, VH, VV-VH ratio/diff, local standard deviation texture).
TRAINING DATA STATUS: Operational in D:\OILTRACE\scripts\preprocessing\prepare_training_data.py. Strict scene-level split (7 scenes train / 448 patches, 2 scenes val / 128 patches, 2 scenes test / 128 patches).
MODEL STATUS: Trained baseline Random Forest (100 estimators, max depth 15) saved at D:\OILTRACE\models\best_model.joblib.
OTB STATUS: C:\OTB (version 9.1.1 / provider 3.0.3) integrated via D:\OILTRACE\scripts\otb_integration.py with ExecutionPolicy bypass.
DETECTION STATUS: Operational. D:\OILTRACE\scripts\training\inference.py generates binary masks & probability maps. D:\OILTRACE\scripts\training\polygonise.py extracts GeoJSON spatial polygons with EPSG:4326 geotransform coordinates, centroid, bounding box, and area in km².
DRIFT & ATTRIBUTION STATUS: Operational. D:\OILTRACE\scripts\attribution\drift_trajectory.py generates 12h Lagrangian hindcast/forecast GeoJSON tracks. D:\OILTRACE\scripts\attribution\vessel_attribution.py produces ranked candidate vessels based on multi-criteria spatial/temporal/vessel-risk scoring.

======================================================================
TESTS PASSED & VERIFIED FILES
======================================================================
- [PASS] D:\OILTRACE\data\dataset_manifest.csv
- [PASS] D:\OILTRACE\data\DATASET_QC_REPORT.txt
- [PASS] D:\OILTRACE\data\processed\features\00000_features.tif (2048x2048x4, float32)
- [PASS] D:\OILTRACE\data\training\train\train_data.npz (448 patches, 256x256x4)
- [PASS] D:\OILTRACE\data\training\validation\validation_data.npz (128 patches)
- [PASS] D:\OILTRACE\data\training\test\test_data.npz (128 patches)
- [PASS] D:\OILTRACE\models\best_model.joblib (83.5 MB)
- [PASS] D:\OILTRACE\outputs\validation\VALIDATION_REPORT.txt
- [PASS] D:\OILTRACE\data\results\detection\00000_spills.geojson (1,222 polygons, 73.6 km² spill)
- [PASS] D:\OILTRACE\data\results\detection\00002_run\00002_spills.geojson (424 polygons, 3.07 km² spill)
- [PASS] D:\OILTRACE\data\results\drift\spill_00000_hindcast.geojson (49 trajectory steps)
- [PASS] D:\OILTRACE\data\results\attribution\ranked_vessel_candidates.csv (4 ranked vessels)
- [PASS] D:\OILTRACE\qgis\OILTRACE.qgz (QGIS 3.44.14 project)

CURRENT BLOCKER:
Incomplete raw image dataset due to upstream Zenodo 504 server timeout.

EXACT NEXT STEP:
Run end-to-end detection on any new SAR scene using:
python D:\OILTRACE\scripts\run_detection_pipeline.py --input <path_to_sar.tif>
======================================================================
