"""
OILTRACE Marine Oil Spill Detection & Attribution Pipeline
----------------------------------------------------------
Production package for autonomous marine oil spill detection in dual polarization Sentinel-1 SAR imagery.
Provides a high precision two stage Object Based Image Analysis (OBIA) processing cascade:
  1. Vessel shadow corridor and bright reflector suppression
  2. Adaptive ocean background damping contrast candidate segmentation
  3. Physical and geometric 17-dimensional object feature extraction
  4. Calibrated Random Forest object classification separating true slicks from lookalikes
"""

from .vessel_masking import (
    detect_vessel_targets,
    mask_vessel_shadows,
    VesselTarget,
    ShadowCorridor,
)
from .segmentation import (
    segment_candidate_dark_spots,
    export_candidate_geojson,
    CandidateDarkSpot,
    SegmentationResult,
)
from .feature_extractor import (
    extract_candidate_features,
    export_features_tabular,
    CandidateObjectFeatures,
    FeatureExtractionResult,
    CANONICAL_FEATURE_NAMES,
)
from .obia_classifier import (
    load_object_classifier,
    ObjectClassifier,
    ObjectClassificationResult,
    TrainedClassifierBundle,
)
from .detect_oil import (
    PipelineRunConfig,
    TwoStagePipelineResult,
    run_two_stage_pipeline,
    resolve_pipeline_mode,
)

__all__ = [
    "detect_vessel_targets",
    "mask_vessel_shadows",
    "VesselTarget",
    "ShadowCorridor",
    "segment_candidate_dark_spots",
    "export_candidate_geojson",
    "CandidateDarkSpot",
    "SegmentationResult",
    "extract_candidate_features",
    "export_features_tabular",
    "CandidateObjectFeatures",
    "FeatureExtractionResult",
    "CANONICAL_FEATURE_NAMES",
    "load_object_classifier",
    "ObjectClassifier",
    "ObjectClassificationResult",
    "TrainedClassifierBundle",
    "PipelineRunConfig",
    "TwoStagePipelineResult",
    "run_two_stage_pipeline",
    "resolve_pipeline_mode",
]
