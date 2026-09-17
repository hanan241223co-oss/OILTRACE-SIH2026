# D:\OILTRACE\scripts\run_detection_pipeline.py
'''
OILTRACE Master Detection Pipeline
Executes end-to-end:
  validate -> preprocess -> features -> inference -> polygonisation -> metrics summary
'''

import os
import sys
import json
import argparse
import numpy as np
import tifffile

# Add local script dirs to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, 'preprocessing'))
sys.path.append(os.path.join(SCRIPT_DIR, 'training'))

from preprocess_sar import preprocess_image
from feature_engineering import compute_features
from inference import run_inference
from polygonise import polygonise

def run_pipeline(sar_input_path, output_dir, model_path=None, threshold=0.3, min_px=50):
    print('======================================================================')
    print('OILTRACE AUTOMATED SAR OIL SPILL DETECTION PIPELINE')
    print('======================================================================')
    print(f'Input SAR Scene: {sar_input_path}')
    print(f'Output Directory: {output_dir}')

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(sar_input_path))[0]

    # 1. Validation & Input Inspection
    print('\n[Step 1/5] Validating Input GeoTIFF...')
    with tifffile.TiffFile(sar_input_path) as tif:
        s = tif.series[0]
        shape = s.shape
        dtype = s.dtype
        print(f'  Raster dimensions: {shape}, dtype: {dtype}')
        if shape[:2] != (2048, 2048):
            print(f'  Warning: Expected (2048, 2048), got {shape[:2]}')

    # 2. Preprocessing & Radiometric Handling
    print('\n[Step 2/5] Preprocessing SAR Imagery (Lee Despeckle & Normalization)...')
    prep_tif = os.path.join(output_dir, f'{base_name}_preprocessed.tif')
    prep_data = preprocess_image(sar_input_path, prep_tif, apply_despeckle=True)
    print(f'  Preprocessed raster saved -> {prep_tif}')

    # 3. Polarimetric & Texture Feature Engineering
    print('\n[Step 3/5] Computing Multi-Polarization Features (VV, VH, VV/VH Ratio, Texture)...')
    feat_tif = os.path.join(output_dir, f'{base_name}_features.tif')
    feat_data = compute_features(sar_input_path, output_dir)
    print(f'  Feature stack saved -> {feat_tif} (Bands: {feat_data.shape[-1]})')

    # 4. Supervised Model Inference
    if model_path is None:
        model_path = r'D:\OILTRACE\models\best_model.joblib'
    print(f'\n[Step 4/5] Running Model Inference using {model_path}...')
    pred_mask_tif = os.path.join(output_dir, f'{base_name}_pred_mask.tif')
    pred_mask, proba_map = run_inference(sar_input_path, model_path, pred_mask_tif, threshold=threshold)
    print(f'  Detection mask saved -> {pred_mask_tif}')

    # 5. Connected Components & GeoJSON Polygonisation
    print('\n[Step 5/5] Extracting Spatial Spill Geometries and Calculating Marine Area...')
    spill_geojson = os.path.join(output_dir, f'{base_name}_spills.geojson')
    spill_features = polygonise(pred_mask_tif, spill_geojson, source_sar_path=sar_input_path, min_pixels=min_px)

    total_spill_area = sum(f['properties']['area_km2'] or 0.0 for f in spill_features)
    print('\n======================================================================')
    print('DETECTION PIPELINE EXECUTION SUMMARY')
    print('======================================================================')
    print(f'Total Spill Detections (>= {min_px} px): {len(spill_features)}')
    print(f'Total Estimated Spill Surface Area: {total_spill_area:.2f} km^2')
    print(f'GeoJSON Spatial Polygon Vector: {spill_geojson}')
    print('======================================================================\n')

    return {
        'preprocessed': prep_tif,
        'features': feat_tif,
        'mask': pred_mask_tif,
        'geojson': spill_geojson,
        'count': len(spill_features),
        'total_area_km2': total_spill_area
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run OILTRACE Master Detection Pipeline')
    parser.add_argument('--input', required=True, help='Path to Sentinel-1 SAR input GeoTIFF')
    parser.add_argument('--output_dir', default=r'D:\OILTRACE\data\results\detection', help='Target output directory')
    parser.add_argument('--model', default=r'D:\OILTRACE\models\best_model.joblib', help='Path to trained model')
    parser.add_argument('--threshold', type=float, default=0.3, help='Classification probability threshold')
    parser.add_argument('--min_px', type=int, default=50, help='Minimum connected pixel area threshold')
    args = parser.parse_args()

    run_pipeline(args.input, args.output_dir, model_path=args.model, threshold=args.threshold, min_px=args.min_px)
