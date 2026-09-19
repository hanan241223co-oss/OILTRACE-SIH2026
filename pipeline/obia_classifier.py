#!/usr/bin/env python3
"""
OILTRACE Object-Based Image Analysis (OBIA) Classifier Engine
------------------------------------------------------------
Implements the runtime inference interface, model bundle serialization,
and decision thresholding for object-level oil spill classification.
Consumes 17 canonical physical, geometric, boundary, and proximity features
to classify candidate dark spot polygons as confirmed slicks or natural lookalikes.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import joblib
import numpy as np

try:
    from pipeline.feature_extractor import CANONICAL_FEATURE_NAMES
except ImportError:
    try:
        from .feature_extractor import CANONICAL_FEATURE_NAMES
    except ImportError:
        from feature_extractor import CANONICAL_FEATURE_NAMES


@dataclass
class ObjectClassificationResult:
    """Classification outcome for a single candidate dark spot polygon."""
    candidate_id: str
    predicted_label: int  # 1 for confirmed spill, 0 for suppressed lookalike
    confidence: float  # probability of mineral oil spill in [0.0, 1.0]
    decision_threshold: float  # threshold used for classification
    classification_status: str  # "confirmed_slick" or "lookalike_suppressed"


@dataclass
class TrainedClassifierBundle:
    """Serialized model bundle containing classifier and metadata."""
    model: Any
    feature_names: List[str]
    decision_threshold: float
    created_at: str
    metrics: Dict[str, float]
    training_manifest_summary: Dict[str, int]
    feature_medians: np.ndarray


class ObjectClassifier:
    """
    Inference engine evaluating candidate dark spots with calibrated
    Random Forest probability estimation and operating thresholding.
    """

    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        decision_threshold: float = 0.60,
        feature_medians: Optional[np.ndarray] = None,
        metrics: Optional[Dict[str, float]] = None,
        training_manifest_summary: Optional[Dict[str, int]] = None,
        created_at: Optional[str] = None
    ) -> None:
        self.model = model
        self.feature_names = feature_names if feature_names is not None else list(CANONICAL_FEATURE_NAMES)
        self.decision_threshold = float(decision_threshold)
        self.metrics = metrics or {}
        self.training_manifest_summary = training_manifest_summary or {}
        self.created_at = created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if feature_medians is not None:
            self.feature_medians = np.array(feature_medians, dtype=np.float32)
        else:
            self.feature_medians = np.zeros(len(self.feature_names), dtype=np.float32)

    def _impute_features(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Replaces infinities and NaNs with training medians (fallback to 0.0).
        """
        mat = np.array(feature_matrix, dtype=np.float32, copy=True)
        # Convert infinities to NaN
        mat[np.isinf(mat)] = np.nan

        # Impute column by column
        for col_idx in range(mat.shape[1]):
            nan_mask = np.isnan(mat[:, col_idx])
            if np.any(nan_mask):
                fill_val = 0.0
                if col_idx < len(self.feature_medians):
                    med = self.feature_medians[col_idx]
                    if not np.isnan(med) and not np.isinf(med):
                        fill_val = float(med)
                mat[nan_mask, col_idx] = fill_val
        return mat

    def predict_features(
        self,
        feature_matrix: np.ndarray,
        decision_threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates a 2D feature matrix of shape (N, 17), returning binary labels
        and probability scores for each candidate.

        Parameters
        ----------
        feature_matrix : np.ndarray
            2D array of shape (N, 17) matching canonical feature ordering.
        decision_threshold : float, optional
            Operating cutoff override. Defaults to self.decision_threshold.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (labels, probabilities) as 1D NumPy arrays of shape (N,).
        """
        mat = np.asarray(feature_matrix)
        if mat.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got array with shape {mat.shape}")
        if mat.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Feature matrix column count ({mat.shape[1]}) does not match "
                f"expected feature count ({len(self.feature_names)})."
            )

        if mat.shape[0] == 0:
            return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32)

        # Impute missing values
        clean_mat = self._impute_features(mat)

        threshold = self.decision_threshold if decision_threshold is None else float(decision_threshold)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(clean_mat)[:, 1].astype(np.float32)
        else:
            probs = self.model.predict(clean_mat).astype(np.float32)

        labels = (probs >= threshold).astype(np.int32)
        return labels, probs

    def predict_candidates(
        self,
        candidates: List[Any],
        decision_threshold: Optional[float] = None
    ) -> List[ObjectClassificationResult]:
        """
        Evaluates a collection of CandidateDarkSpot objects, populates their
        in-place classification attributes, and returns structured result entities.

        Parameters
        ----------
        candidates : List[CandidateDarkSpot]
            List of candidate objects from segmentation / feature extraction.
        decision_threshold : float, optional
            Operating cutoff override. Defaults to self.decision_threshold.

        Returns
        -------
        List[ObjectClassificationResult]
            Classification decisions for each candidate spot.
        """
        if not candidates:
            return []

        feature_vectors = []
        for cand in candidates:
            if hasattr(cand, "features") and cand.features is not None:
                feat_obj = cand.features
                if hasattr(feat_obj, "to_numpy_vector"):
                    vec = feat_obj.to_numpy_vector()
                elif hasattr(feat_obj, "feature_vector") and isinstance(feat_obj.feature_vector, dict):
                    vec = [feat_obj.feature_vector.get(name, 0.0) for name in self.feature_names]
                else:
                    vec = [getattr(feat_obj, name, 0.0) for name in self.feature_names]
            elif hasattr(cand, "feature_vector") and isinstance(cand.feature_vector, dict):
                vec = [cand.feature_vector.get(name, 0.0) for name in self.feature_names]
            elif hasattr(cand, "feature_vector") and isinstance(cand.feature_vector, (list, np.ndarray)):
                vec = list(cand.feature_vector)
            else:
                # Direct attribute lookup on CandidateDarkSpot
                vec = [getattr(cand, name, 0.0) for name in self.feature_names]

            feature_vectors.append(vec)

        feat_matrix = np.array(feature_vectors, dtype=np.float32)
        labels, probs = self.predict_features(feat_matrix, decision_threshold=decision_threshold)

        threshold = self.decision_threshold if decision_threshold is None else float(decision_threshold)
        results = []

        for i, cand in enumerate(candidates):
            lbl = int(labels[i])
            prob = float(probs[i])
            status = "confirmed_slick" if lbl == 1 else "lookalike_suppressed"

            # Populate in-place attributes if candidate supports them
            if hasattr(cand, "is_confirmed_slick"):
                cand.is_confirmed_slick = (lbl == 1)
            if hasattr(cand, "slick_probability"):
                cand.slick_probability = prob

            cid = getattr(cand, "candidate_id", f"candidate_{i+1:03d}")
            res = ObjectClassificationResult(
                candidate_id=cid,
                predicted_label=lbl,
                confidence=prob,
                decision_threshold=threshold,
                classification_status=status
            )
            results.append(res)

        return results


def save_object_classifier(bundle: TrainedClassifierBundle, model_path: str) -> None:
    """
    Serializes a TrainedClassifierBundle to disk via joblib.
    """
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    bundle_dict = {
        "model": bundle.model,
        "feature_names": list(bundle.feature_names),
        "decision_threshold": float(bundle.decision_threshold),
        "created_at": bundle.created_at,
        "metrics": dict(bundle.metrics),
        "training_manifest_summary": dict(bundle.training_manifest_summary),
        "feature_medians": np.array(bundle.feature_medians, dtype=np.float32)
    }
    joblib.dump(bundle_dict, model_path)


def load_object_classifier(model_path: str) -> ObjectClassifier:
    """
    Loads and validates an ObjectClassifier from a serialized joblib bundle.
    Raises FileNotFoundError if model_path does not exist, or ValueError
    if bundle metadata or feature schemas are corrupted.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    data = joblib.load(model_path)

    if isinstance(data, dict):
        required_keys = ["model", "feature_names"]
        for k in required_keys:
            if k not in data:
                raise ValueError(f"Malformed model bundle: missing required key '{k}'")

        feature_names = data["feature_names"]
        if feature_names != CANONICAL_FEATURE_NAMES:
            missing = set(CANONICAL_FEATURE_NAMES) - set(feature_names)
            extra = set(feature_names) - set(CANONICAL_FEATURE_NAMES)
            raise ValueError(
                f"Feature schema mismatch in model bundle. Missing: {missing}, Extra: {extra}"
            )

        threshold = data.get("decision_threshold", 0.60)
        feature_medians = data.get("feature_medians", None)
        metrics = data.get("metrics", {})
        summary = data.get("training_manifest_summary", {})
        created_at = data.get("created_at", None)

        return ObjectClassifier(
            model=data["model"],
            feature_names=feature_names,
            decision_threshold=threshold,
            feature_medians=feature_medians,
            metrics=metrics,
            training_manifest_summary=summary,
            created_at=created_at
        )
    elif hasattr(data, "predict_proba") or hasattr(data, "predict"):
        # Direct raw classifier fallback
        return ObjectClassifier(
            model=data,
            feature_names=list(CANONICAL_FEATURE_NAMES),
            decision_threshold=0.60
        )
    else:
        raise ValueError(f"Unrecognized object classifier artifact type: {type(data)}")
