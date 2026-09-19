# 0005. Two Stage Pipeline Verification in Testing · Verification Plan

This document details executable verification commands and validation criteria to confirm all acceptance criteria (AC-1 through AC-5) for Feature 12.

## Verification Checklist

- [x] Step 1: CLI Interface & Help Inspection (AC-1)
- [x] Step 2: Two Stage Detection on Reference Scene (AC-1, AC-2, AC-4)
- [x] Step 3: Verify Output Deliverables Structure & Content (AC-2, AC-4)
- [x] Step 4: Verify Decision Threshold Override (AC-3)
- [x] Step 5: Verify Calm Sea Clutter / Zero Candidate Handling (AC-5)
- [x] Step 6: Verify Backward Compatibility for Legacy Models (AC-1)
- [x] Step 7: Automated Test Suite Execution (AC-1 through AC-5)

---

### Step 1: CLI Interface & Help Inspection (AC-1)

Verify that `testing/detect_oil.py` presents the `--mode` option with supported choices and proper documentation.

```bash
python testing/detect_oil.py --help
```

**Expected output**:
* CLI options list includes `--mode {two_stage,unet,rf_pixel}` with default behavior clearly described.
* CLI options list includes `--threshold` and `--obia-model`.
* Command exits with status code 0.

---

### Step 2: Two Stage Detection on Reference Scene (AC-1, AC-2, AC-4)

Execute the complete two stage detection pipeline on canonical test scene `testing/det_20260917163850_004aae`.

```bash
python testing/detect_oil.py \
  --input testing/det_20260917163850_004aae/scene.tif \
  --output testing/det_20260917163850_004aae \
  --mode two_stage
```

**Expected output**:
* Pipeline executes sequential cascade: vessel masking, candidate segmentation, feature extraction, and OBIA classification.
* Console logs report vessel counts, candidate counts, and confirmed slick tally.
* Deliverables written:
  * `testing/det_20260917163850_004aae/scene_spills.geojson`
  * `testing/det_20260917163850_004aae/mask.png`
  * `testing/det_20260917163850_004aae/result.json`
  * `testing/det_20260917163850_004aae/scene_candidates.geojson`
* Command exits with status code 0.

---

### Step 3: Verify Output Deliverables Structure & Content (AC-2, AC-4)

Verify that the generated deliverable files conform to expected schemas and spatial criteria.

```bash
python -c "
import json
import numpy as np
from PIL import Image

# 1. Verify result.json
with open('testing/det_20260917163850_004aae/result.json') as f:
    res = json.load(f)
assert res['pipeline_mode'] == 'two_stage', 'Mode mismatch in result.json'
assert res['object_classification']['confirmed_slicks'] >= 1, 'No confirmed slicks reported'
print(f'Confirmed slicks in result.json: {res[\"object_classification\"][\"confirmed_slicks\"]}')

# 2. Verify mask.png dimensions and values
mask = np.array(Image.open('testing/det_20260917163850_004aae/mask.png'))
assert mask.shape == (2048, 2048), f'Mask shape mismatch: {mask.shape}'
unique_vals = set(np.unique(mask))
assert unique_vals.issubset({0, 255}), f'Unexpected mask pixel values: {unique_vals}'
assert 255 in unique_vals, 'No positive slick pixels found in mask.png'
print(f'Mask verification passed: shape {mask.shape}, positive pixels: {np.sum(mask == 255)}')

# 3. Verify scene_spills.geojson
with open('testing/det_20260917163850_004aae/scene_spills.geojson') as f:
    spills = json.load(f)
assert spills['type'] == 'FeatureCollection', 'Invalid GeoJSON type'
assert len(spills['features']) >= 1, 'No features in scene_spills.geojson'
props = spills['features'][0]['properties']
for required_prop in ['candidate_id', 'confidence', 'area_km2', 'classification_status']:
    assert required_prop in props, f'Missing property: {required_prop}'
assert props['classification_status'] == 'confirmed_slick'
print(f'Spills GeoJSON verified: {len(spills[\"features\"])} slick features found.')
"
```

---

### Step 4: Verify Decision Threshold Override (AC-3)

Test runtime probability cutoff overrides using `--threshold`.

```bash
# High threshold: should suppress borderline detections
python testing/detect_oil.py \
  --input testing/det_20260917163850_004aae/scene.tif \
  --output testing/det_20260917163850_004aae \
  --mode two_stage \
  --threshold 0.999
```

**Expected output**:
* Pipeline executes successfully.
* With a very high threshold (0.999), 0 slicks are confirmed.
* `mask.png` is all zero black, and `scene_spills.geojson` has 0 features.

---

### Step 5: Verify Calm Sea Clutter / Zero Candidate Handling (AC-5)

Verify that processing a synthetic SAR raster with no dark spot candidates executes cleanly without raising exceptions.

```bash
python -c "
import tempfile
import os
import tifffile
import numpy as np
from testing.detect_oil import run_two_stage_pipeline, PipelineRunConfig

# Create uniform high backscatter synthetic raster (no dark spots)
high_backscatter = np.full((2, 256, 256), -5.0, dtype=np.float32)

with tempfile.TemporaryDirectory() as tmpdir:
    scene_path = os.path.join(tmpdir, 'scene.tif')
    tifffile.imwrite(scene_path, high_backscatter)
    
    config = PipelineRunConfig(
        input_path=scene_path,
        output_dir=tmpdir,
        mode='two_stage',
        obia_model_path='testing/best_obia_rf.joblib',
        unet_model_path='testing/best_unet.pt',
        pixel_rf_path='testing/best_model.joblib',
        threshold=0.60,
        enable_vessel_masking=True,
        enable_candidate_export=True,
        device='cpu'
    )
    res = run_two_stage_pipeline(config)
    assert res.confirmed_slicks_count == 0, 'Expected 0 confirmed slicks on uniform raster'
    assert os.path.exists(res.deliverables['mask_png']), 'mask.png missing'
    assert os.path.exists(res.deliverables['spills_geojson']), 'spills_geojson missing'
    assert os.path.exists(res.deliverables['result_json']), 'result_json missing'
    print('Zero candidate calm water edge case test passed successfully.')
"
```

---

### Step 6: Verify Backward Compatibility for Legacy Models (AC-1)

Verify that running `--model testing/best_unet.pt` and `--model testing/best_model.joblib` continue to function as expected.

```bash
# U-Net backward compatibility
python testing/detect_oil.py \
  --input testing/det_20260917163850_004aae/scene.tif \
  --output testing/det_20260917163850_004aae \
  --model testing/best_unet.pt

# Pixel Random Forest baseline backward compatibility
python testing/detect_oil.py \
  --input testing/det_20260917163850_004aae/scene.tif \
  --output testing/det_20260917163850_004aae \
  --model testing/best_model.joblib
```

**Expected output**:
* Both commands execute without error and generate valid detection outputs.

---

### Step 7: Automated Test Suite Execution (AC-1 through AC-5)

Run the automated test suite to ensure all unit and integration tests pass.

```bash
pytest -q testing/test_detect_oil.py
pytest -q testing/
```

**Expected output**:
* 100% of tests pass across all suites.
