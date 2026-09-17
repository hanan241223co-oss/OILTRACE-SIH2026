# D:\OILTRACE\scripts\preprocessing\prepare_training_data.py
'''
OILTRACE Training Data Preparation Pipeline
Extracts non-overlapping (or overlapping) patches from SAR scenes & masks.
Enforces scene-level split (NO pixel-level leakage across scenes).
Saves training-ready .npz or .tif tiles under train/validation/test.
'''

import os
import csv
import argparse
import numpy as np
import tifffile
from feature_engineering import compute_features

def generate_patches(img_stack, mask_arr, patch_size=256, stride=256):
    H, W, C = img_stack.shape
    patches_img = []
    patches_mask = []

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch_i = img_stack[y:y+patch_size, x:x+patch_size, :]
            patch_m = mask_arr[y:y+patch_size, x:x+patch_size]
            patches_img.append(patch_i)
            patches_mask.append(patch_m)

    return np.array(patches_img, dtype=np.float32), np.array(patches_mask, dtype=np.uint8)

def build_dataset(manifest_csv, output_root, patch_size=256, stride=256):
    # Read manifest for valid scenes
    valid_oil = []
    valid_lookalike = []
    valid_no_oil = []

    with open(manifest_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['valid'].lower() == 'true':
                if row['class'] == 'oil_spill':
                    valid_oil.append(row)
                elif row['class'] == 'look_alike':
                    valid_lookalike.append(row)
                elif row['class'] == 'no_oil':
                    valid_no_oil.append(row)

    print(f'Found valid scenes: {len(valid_oil)} oil_spill, {len(valid_lookalike)} look_alike, {len(valid_no_oil)} no_oil')

    # Scene-level split:
    # 9 oil scenes: 6 train, 2 validation, 1 test
    # 2 lookalike scenes: 1 train, 1 test
    splits = {
        'train': valid_oil[:6] + valid_lookalike[:1],
        'validation': valid_oil[6:8],
        'test': valid_oil[8:] + valid_lookalike[1:]
    }

    stats = {}
    for split_name, scenes in splits.items():
        split_dir = os.path.join(output_root, split_name)
        os.makedirs(split_dir, exist_ok=True)
        split_imgs = []
        split_masks = []
        oil_pixel_count = 0
        bg_pixel_count = 0

        for sc in scenes:
            ip = sc['image_path']
            mp = sc['mask_path']
            # Compute 4-band features
            feats = compute_features(ip)
            with tifffile.TiffFile(mp) as tif:
                mask = tif.asarray()
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                mask = (mask > 0).astype(np.uint8)

            p_imgs, p_masks = generate_patches(feats, mask, patch_size=patch_size, stride=stride)
            split_imgs.append(p_imgs)
            split_masks.append(p_masks)
            oil_pixel_count += np.sum(p_masks == 1)
            bg_pixel_count += np.sum(p_masks == 0)

        all_imgs = np.concatenate(split_imgs, axis=0) if split_imgs else np.zeros((0, patch_size, patch_size, 4))
        all_masks = np.concatenate(split_masks, axis=0) if split_masks else np.zeros((0, patch_size, patch_size))

        out_npz = os.path.join(split_dir, f'{split_name}_data.npz')
        np.savez_compressed(out_npz, images=all_imgs, masks=all_masks)
        stats[split_name] = {
            'scenes': len(scenes),
            'patches': len(all_imgs),
            'oil_pixels': int(oil_pixel_count),
            'bg_pixels': int(bg_pixel_count),
            'file': out_npz
        }
        print(f'[{split_name}] {len(scenes)} scenes -> {len(all_imgs)} patches ({patch_size}x{patch_size}). Saved to {out_npz}')

    return stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default=r'D:\OILTRACE\data\dataset_manifest.csv')
    parser.add_argument('--output', default=r'D:\OILTRACE\data\training')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=256)
    args = parser.parse_args()

    s = build_dataset(args.manifest, args.output, patch_size=args.patch_size, stride=args.stride)
