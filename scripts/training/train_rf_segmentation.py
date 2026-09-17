# D:\OILTRACE\scripts\training\train_rf_segmentation.py
'''
OILTRACE Random Forest / Lightweight Pixel Classifier Baseline
Trained on 4-band SAR features (VV, VH, VV-VH diff, Texture)
Pixel-level semantic segmentation model using scikit-learn RandomForestClassifier.
'''

import os
import argparse
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, jaccard_score, f1_score, precision_score, recall_score

def train_model(train_npz, val_npz, model_output, n_estimators=100, max_depth=15, sample_rate=0.05, random_state=42):
    print(f'Loading training data from {train_npz}...')
    tr_data = np.load(train_npz)
    X_tr = tr_data['images'] # (N, 256, 256, 4)
    y_tr = tr_data['masks']  # (N, 256, 256)

    print(f'Loading validation data from {val_npz}...')
    val_data = np.load(val_npz)
    X_val = val_data['images']
    y_val = val_data['masks']

    # Flatten to pixel table
    N_tr, H, W, C = X_tr.shape
    X_tr_flat = X_tr.reshape(-1, C)
    y_tr_flat = y_tr.reshape(-1)

    # Subsample to keep memory reasonable and balance background/oil
    oil_indices = np.where(y_tr_flat == 1)[0]
    bg_indices = np.where(y_tr_flat == 0)[0]

    np.random.seed(random_state)
    # Balance classes: take all oil pixels (or max 100k) and 3x bg pixels
    n_oil = min(len(oil_indices), 100000)
    selected_oil = np.random.choice(oil_indices, size=n_oil, replace=False) if len(oil_indices) > 0 else np.array([])
    selected_bg = np.random.choice(bg_indices, size=min(len(bg_indices), max(n_oil * 3, 200000)), replace=False)

    sample_idx = np.concatenate([selected_oil, selected_bg]).astype(int)
    np.random.shuffle(sample_idx)

    X_train_sub = X_tr_flat[sample_idx]
    y_train_sub = y_tr_flat[sample_idx]

    print(f'Training Random Forest on {len(X_train_sub):,} pixel samples (Oil: {len(selected_oil):,}, BG: {len(selected_bg):,})...')
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
        class_weight='balanced_subsample'
    )
    clf.fit(X_train_sub, y_train_sub)

    # Validation evaluation
    X_val_flat = X_val.reshape(-1, C)
    y_val_flat = y_val.reshape(-1)

    print(f'Evaluating on validation set ({len(X_val_flat):,} pixels)...')
    y_val_pred = clf.predict(X_val_flat)

    prec = precision_score(y_val_flat, y_val_pred, zero_division=0)
    rec = recall_score(y_val_flat, y_val_pred, zero_division=0)
    f1 = f1_score(y_val_flat, y_val_pred, zero_division=0)
    iou = jaccard_score(y_val_flat, y_val_pred, zero_division=0)

    print('=== Validation Results ===')
    print(f'Precision: {prec:.4f}')
    print(f'Recall:    {rec:.4f}')
    print(f'F1-Score:  {f1:.4f}')
    print(f'IoU:       {iou:.4f}')

    # Save model
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump(clf, model_output)
    print(f'Saved trained model to {model_output}')

    # Save report
    report_path = os.path.join(r'D:\OILTRACE\data\training', 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write(f'''OILTRACE Random Forest Segmentation Training Report
Generated: 2026-09-16
Model: RandomForestClassifier (n_estimators={n_estimators}, max_depth={max_depth})
Features: 4 channels (VV_norm, VH_norm, VV_VH_diff, VV_texture)
Training Samples: {len(X_train_sub)} pixels (Oil: {len(selected_oil)}, BG: {len(selected_bg)})
Validation Results:
  - Precision: {prec:.4f}
  - Recall:    {rec:.4f}
  - F1-Score:  {f1:.4f}
  - IoU:       {iou:.4f}
Classification Report:
{classification_report(y_val_flat, y_val_pred, digits=4, zero_division=0)}
''')
    print(f'Report written to {report_path}')
    return clf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_npz', default=r'D:\OILTRACE\data\training\train\train_data.npz')
    parser.add_argument('--val_npz', default=r'D:\OILTRACE\data\training\validation\validation_data.npz')
    parser.add_argument('--model_out', default=r'D:\OILTRACE\models\best_model.joblib')
    parser.add_argument('--trees', type=int, default=100)
    parser.add_argument('--depth', type=int, default=15)
    args = parser.parse_args()

    train_model(args.train_npz, args.val_npz, args.model_out, n_estimators=args.trees, max_depth=args.depth)
