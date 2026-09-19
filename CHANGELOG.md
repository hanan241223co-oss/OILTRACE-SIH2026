# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
* Adaptive dark spot candidate segmentation using moving window ambient ocean clutter convolution (see spec 0002)
* Relative damping ratio contrast calculation with automatic fallback to global ocean median in low sample coastal zones
* Morphological closing and 8-connectivity connected component labeling to bridge interior slick voids and prune speckle noise
* Vessel proximity linkage to identify suspect discharging vessels within 35 pixels while preserving true slicks
* High performance vectorized polygon extraction via rasterio shapes and geodesic area calculations via pyproj in square kilometers
* Automated unit and integration test suite in `testing/test_segmentation.py` with 15 tests

### Changed
* Integrated candidate segmentation into the operational detection pipeline in `testing/detect_oil.py` as Step 3b
* Added command line options for `--min_damping`, `--max_backscatter`, `--window_size`, `--min_candidate_px`, and `--no_candidate_segmentation`
* Automated export of candidate polygons to GeoJSON alongside final detection deliverables

## [0.2.0] - 2026-09-19

### Added
* Autonomous metallic vessel reflector detector using dual polarization peak and local contrast filtering (see spec 0001)
* Directional radar shadow corridor projection with radial buffer fallback for vessels without orbital metadata
* Physical damping ratio contrast check and boundary Sobel gradient validation to preserve true oil slicks near vessels
* Automated GeoJSON FeatureCollection export of detected vessels with geographic positions and suspect polluter flags
* Unit and integration test suite in `testing/test_vessel_masking.py` with 10 automated tests

### Changed
* Integrated vessel and bright target shadow masking into operational detection pipeline as Step 3
* Added `--no_vessel_masking` command line argument to toggle vessel shadow suppression
