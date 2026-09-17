# D:\OILTRACE\scripts\training\evaluate_model.py
'''
OILTRACE Model Evaluation on Held-Out Test Set
Computes Precision, Recall, F1, IoU, confusion matrix and saves report.
'''

import os
import argparse
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, jaccard_score

def evaluate(model_path, test_npz, output_report):
    print(f'Loading model: {model_path}')
    clf = joblib.load(model_path)

    print(f'Loading test data: {test_npz}')
    test_data = np.load(test_npz)
    X_test = test_data['images']
    y_test = test_data['masks']

    N, H, W, C = X_test.shape
    X_flat = X_test.reshape(-1, C)
    y_flat = y_test.reshape(-1)

    print(f'Evaluating on {len(X_flat):,} test pixels...')
    y_pred = clf.predict(X_flat)

    prec = precision_score(y_flat, y_pred, zero_division=0)
    rec = recall_score(y_flat, y_pred, zero_division=0)
    f1 = f1_score(y_flat, y_pred, zero_division=0)
    iou = jaccard_score(y_flat, y_pred, zero_division=0)
    cm = confusion_matrix(y_flat, y_pred)

    print('=== Test Evaluation Metrics ===')
    print(f'Precision: {prec:.4f}')
    print(f'Recall:    {rec:.4f}')
    print(f'F1-Score:  {f1:.4f}')
    print(f'IoU:       {iou:.4f}')
    print('Confusion Matrix:\n', cm)

    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, 'w') as f:
        f.write(f'''OILTRACE TEST EVALUATION REPORT
Model: {model_path}
Test dataset: {test_npz}
Total pixels evaluated: {len(X_flat):,}
Oil pixels in ground truth: {int(np.sum(y_flat)):,} ({(np.sum(y_flat)/len(y_flat))*100:.2f}%)
Predicted oil pixels: {int(np.sum(y_pred)):,}

Metrics (Positive Class: Oil Spill):
  - Precision: {prec:.4f}
  - Recall:    {rec:.4f}
  - F1-Score:  {f1:.4f}
  - IoU:       {iou:.4f}

Confusion Matrix (TN, FP / FN, TP):
{cm}

Full Classification Report:
{classification_report(y_flat, y_pred, digits=4, zero_division=0)}
''')
    print(f'Test evaluation saved to {output_report}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=r'D:\OILTRACE\models\best_model.joblib')
    parser.add_argument('--test_npz', default=r'D:\OILTRACE\data\training\test\test_data.npz')
    parser.add_argument('--output', default=r'D:\OILTRACE\outputs\validation\VALIDATION_REPORT.txt')
    args = parser.parse_args()

    evaluate(args.model, args.test_npz, args.output)
