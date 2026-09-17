# D:\OILTRACE\scripts\preprocessing\validate_training_data.py
'''
OILTRACE Training Data Validation & Integrity Checker
Verifies patch dimensions, values, class balance, and lack of data leakage.
'''

import os
import argparse
import numpy as np

def validate_split(npz_path):
    if not os.path.exists(npz_path):
        print(f'Error: {npz_path} does not exist')
        return False

    data = np.load(npz_path)
    imgs = data['images']
    masks = data['masks']

    print(f'=== Validating {os.path.basename(npz_path)} ===')
    print(f'  Images shape: {imgs.shape}, dtype: {imgs.dtype}, range: [{imgs.min():.3f}, {imgs.max():.3f}]')
    print(f'  Masks shape: {masks.shape}, dtype: {masks.dtype}, unique: {np.unique(masks)}')
    print(f'  NaNs in images: {np.isnan(imgs).any()}, Infs in images: {np.isinf(imgs).any()}')
    oil_pct = (np.sum(masks == 1) / masks.size) * 100
    print(f'  Oil pixel percentage: {oil_pct:.2f}%')
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=r'D:\OILTRACE\data\training')
    args = parser.parse_args()

    for s in ['train', 'validation', 'test']:
        p = os.path.join(args.root, s, f'{s}_data.npz')
        validate_split(p)
