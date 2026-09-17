# D:\OILTRACE\scripts\preprocessing\feature_engineering.py
'''
OILTRACE Feature Engineering Module
Generates discriminative SAR polarization features:
- VV (dB / normalised)
- VH (dB / normalised)
- Ratio: VV / VH (linear) = VV(dB) - VH(dB)
- Difference: VV(dB) - VH(dB) (Cross-polarization ratio in dB)
- Local Texture: Local standard deviation (5x5) capturing slick boundary dampening
'''

import os
import argparse
import numpy as np
import tifffile
from scipy.ndimage import uniform_filter

def compute_features(input_path, output_dir=None):
    with tifffile.TiffFile(input_path) as tif:
        data = tif.asarray().astype(np.float32)

    if data.ndim == 3 and data.shape[0] == 2:
        data = np.transpose(data, (1, 2, 0))

    vv_db = np.clip(data[:, :, 0], -50.0, 5.0)
    vh_db = np.clip(data[:, :, 1], -50.0, 5.0)

    # 1. VV and VH normalised
    vv_norm = np.clip((vv_db - (-45.0)) / 45.0, 0.0, 1.0)
    vh_norm = np.clip((vh_db - (-45.0)) / 45.0, 0.0, 1.0)

    # 2. Polarimetric difference / ratio in dB: (VV_dB - VH_dB)
    # In linear power: sigma0_vv / sigma0_vh. In dB: VV_dB - VH_dB
    diff_db = vv_db - vh_db
    diff_norm = np.clip((diff_db - (-10.0)) / 25.0, 0.0, 1.0)

    # 3. Local standard deviation (Texture feature)
    vv_mean = uniform_filter(vv_norm, (5, 5))
    vv_sqr_mean = uniform_filter(vv_norm**2, (5, 5))
    vv_std = np.sqrt(np.maximum(vv_sqr_mean - vv_mean**2, 0.0))
    vv_std_norm = np.clip(vv_std / 0.15, 0.0, 1.0)

    # Multi-feature stack: 4 channels (VV, VH, VV_VH_diff, VV_texture)
    feature_stack = np.stack([vv_norm, vh_norm, diff_norm, vv_std_norm], axis=-1).astype(np.float32)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(input_path))[0]
        out_path = os.path.join(output_dir, f'{base}_features.tif')
        tifffile.imwrite(out_path, feature_stack)
        print(f'Saved 4-band features to {out_path}')

    return feature_stack

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output_dir', default=r'D:\OILTRACE\data\processed\features')
    args = parser.parse_args()

    feats = compute_features(args.input, args.output_dir)
    print('Computed features shape:', feats.shape, 'dtype:', feats.dtype)
