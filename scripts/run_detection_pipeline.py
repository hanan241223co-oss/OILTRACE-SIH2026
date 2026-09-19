#!/usr/bin/env python3
"""
OILTRACE Master Detection Pipeline
Executes automated satellite SAR oil spill detection and polygonisation.
Integrates the high precision two stage object based image analysis cascade.
"""

import os
import sys
import json
import argparse
import numpy as np
import tifffile

# Add repo root and script subdirectories to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.append(os.path.join(SCRIPT_DIR, 'preprocessing'))
sys.path.append(os.path.join(SCRIPT_DIR, 'training'))

try:
    from pipeline.detect_oil import run_two_stage_pipeline, PipelineRunConfig
    HAS_TWO_STAGE = True
except ImportError:
    HAS_TWO_STAGE = False


def run_pipeline(sar_input_path: str, output_dir: str, model_path: str = None, threshold: float = None, min_px: int = 50):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(sar_input_path))[0]

    # Resolve default model in pipeline/models if not specified
    if model_path is None:
        candidate_model = os.path.join(REPO_ROOT, "pipeline", "models", "best_obia_rf.joblib")
        if os.path.exists(candidate_model):
            model_path = candidate_model

    if HAS_TWO_STAGE and (model_path is None or "obia" in os.path.basename(model_path).lower() or model_path.endswith(".joblib")):
        print("======================================================================")
        print("OILTRACE AUTOMATED SAR OIL SPILL DETECTION PIPELINE (OBIA TWO STAGE)")
        print("======================================================================")
        print(f"Input SAR Scene:  {sar_input_path}")
        print(f"Output Directory: {output_dir}")
        print(f"Model Path:       {model_path}")

        config = PipelineRunConfig(
            input_path=sar_input_path,
            output_dir=output_dir,
            mode="two_stage",
            obia_model_path=model_path,
            threshold=threshold,
            min_px=min_px
        )
        res = run_two_stage_pipeline(config)
        return {
            'mask': res.deliverables['mask_png'],
            'geojson': res.deliverables['spills_geojson'],
            'count': res.confirmed_slicks_count,
            'total_area_km2': res.confirmed_slicks_area_km2,
            'result_json': res.deliverables['result_json']
        }

    # Fallback to legacy pixel pipeline if two stage is unavailable
    from preprocess_sar import preprocess_image
    from feature_engineering import compute_features
    from inference import run_inference
    from polygonise import polygonise

    print("======================================================================")
    print("OILTRACE SAR OIL SPILL DETECTION PIPELINE (PIXEL BASELINE)")
    print("======================================================================")
    print(f"Input SAR Scene:  {sar_input_path}")
    print(f"Output Directory: {output_dir}")

    # 1. Validation & Input Inspection
    print("\n[Step 1/5] Validating Input GeoTIFF...")
    with tifffile.TiffFile(sar_input_path) as tif:
        s = tif.series[0]
        shape = s.shape
        dtype = s.dtype
        print(f"  Raster dimensions: {shape}, dtype: {dtype}")

    # 2. Preprocessing
    print("\n[Step 2/5] Preprocessing SAR Imagery (Lee Despeckle & Normalization)...")
    prep_tif = os.path.join(output_dir, f"{base_name}_preprocessed.tif")
    prep_data = preprocess_image(sar_input_path, prep_tif, apply_despeckle=True)
    print(f"  Preprocessed raster saved -> {prep_tif}")

    # 3. Features
    print("\n[Step 3/5] Computing Multi-Polarization Features...")
    feat_tif = os.path.join(output_dir, f"{base_name}_features.tif")
    feat_data = compute_features(sar_input_path, output_dir)
    print(f"  Feature stack saved -> {feat_tif}")

    # 4. Supervised Model Inference
    eff_threshold = threshold if threshold is not None else 0.3
    print(f"\n[Step 4/5] Running Model Inference using {model_path}...")
    pred_mask_tif = os.path.join(output_dir, f"{base_name}_pred_mask.tif")
    pred_mask, proba_map = run_inference(sar_input_path, model_path, pred_mask_tif, threshold=eff_threshold)
    print(f"  Detection mask saved -> {pred_mask_tif}")

    # 5. Connected Components & GeoJSON Polygonisation
    print("\n[Step 5/5] Extracting Spatial Spill Geometries...")
    spill_geojson = os.path.join(output_dir, f"{base_name}_spills.geojson")
    spill_features = polygonise(pred_mask_tif, spill_geojson, source_sar_path=sar_input_path, min_pixels=min_px)
    total_spill_area = sum(f['properties']['area_km2'] or 0.0 for f in spill_features)

    print("\n======================================================================")
    print("DETECTION PIPELINE EXECUTION SUMMARY")
    print("======================================================================")
    print(f"Total Spill Detections (>= {min_px} px): {len(spill_features)}")
    print(f"Total Estimated Spill Surface Area: {total_spill_area:.2f} km^2")
    print(f"GeoJSON Spatial Polygon Vector: {spill_geojson}")
    print("======================================================================\n")

    return {
        'preprocessed': prep_tif,
        'features': feat_tif,
        'mask': pred_mask_tif,
        'geojson': spill_geojson,
        'count': len(spill_features),
        'total_area_km2': total_spill_area
    }


if __name__ == '__main__':
    default_input = os.path.join(REPO_ROOT, "pipeline", "data", "scene.tif")
    default_output = os.path.join(REPO_ROOT, "pipeline", "outputs")
    default_model = os.path.join(REPO_ROOT, "pipeline", "models", "best_obia_rf.joblib")

    parser = argparse.ArgumentParser(description="Run OILTRACE Master Detection Pipeline")
    parser.add_argument("--input", default=default_input, help=f"Path to Sentinel-1 SAR input GeoTIFF (default: {default_input})")
    parser.add_argument("--output_dir", default=default_output, help=f"Target output directory (default: {default_output})")
    parser.add_argument("--model", default=default_model, help=f"Path to trained model (default: {default_model})")
    parser.add_argument("--threshold", type=float, default=None, help="Classification probability threshold")
    parser.add_argument("--min_px", type=int, default=50, help="Minimum connected pixel area threshold (default: 50)")
    args = parser.parse_args()

    run_pipeline(args.input, args.output_dir, model_path=args.model, threshold=args.threshold, min_px=args.min_px)
