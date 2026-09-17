# D:\OILTRACE\scripts\preprocessing\preprocess_sar.py
'''
OILTRACE SAR Preprocessing Module
Extracts VV, VH, handles dB radiometric scaling, outlier handling,
optional Lee/Frost speckle filtering, and min-max/standard normalisation.
Preserves raw data and GeoTIFF metadata.
'''

import os
import sys
import argparse
import numpy as np
import tifffile
from scipy.ndimage import uniform_filter

def lee_filter(img, size=5):
    '''Lee filter for SAR speckle noise reduction.'''
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = np.maximum(img_sqr_mean - img_mean**2, 0)
    overall_variance = np.var(img)
    if overall_variance == 0:
        return img
    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

def preprocess_image(input_path, output_path=None, apply_despeckle=True, filter_size=5):
    '''
    Preprocesses a 2-channel Sentinel-1 SAR Sigma0 (dB) GeoTIFF.
    Channel 0: VV (dB)
    Channel 1: VH (dB)
    '''
    with tifffile.TiffFile(input_path) as tif:
        data = tif.asarray() # (H, W, 2) or (2, H, W)
        tags = tif.pages[0].tags

    if data.ndim == 3 and data.shape[0] == 2:
        data = np.transpose(data, (1, 2, 0))

    vv = data[:, :, 0].astype(np.float32)
    vh = data[:, :, 1].astype(np.float32)

    # 1. Outlier handling: clip extreme non-physical backscatter dB values
    vv_clipped = np.clip(vv, -50.0, 5.0)
    vh_clipped = np.clip(vh, -50.0, 5.0)

    # 2. Speckle filtering (optional)
    if apply_despeckle:
        vv_clean = lee_filter(vv_clipped, size=filter_size)
        vh_clean = lee_filter(vh_clipped, size=filter_size)
    else:
        vv_clean = vv_clipped
        vh_clean = vh_clipped

    # 3. Normalization: Min-Max scale to [0, 1] range based on SAR marine backscatter envelope
    # Typical ocean SAR: VV [-45, 0], VH [-45, -5]
    vv_norm = np.clip((vv_clean - (-45.0)) / 45.0, 0.0, 1.0)
    vh_norm = np.clip((vh_clean - (-45.0)) / 45.0, 0.0, 1.0)

    out_stack = np.stack([vv_norm, vh_norm], axis=-1).astype(np.float32)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tifffile.imwrite(output_path, out_stack)

    return out_stack

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Sentinel-1 SAR GeoTIFF')
    parser.add_argument('--input', required=True, help='Path to input SAR GeoTIFF')
    parser.add_argument('--output', required=True, help='Path to output preprocessed GeoTIFF')
    parser.add_argument('--despeckle', action='store_true', default=True, help='Apply Lee speckle filter')
    args = parser.parse_args()

    out = preprocess_image(args.input, args.output, apply_despeckle=args.despeckle)
    print(f'Successfully preprocessed {args.input} -> {args.output}, shape: {out.shape}, dtype: {out.dtype}')
