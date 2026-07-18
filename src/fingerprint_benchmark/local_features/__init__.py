"""Shared detector-neutral local-feature pipeline components."""

from .detector_only import (
    PROTOCOL_NAME,
    DetectorOnlyAdapter,
    DetectorOnlyProtocolConfig,
    adapt_detector_result,
    build_representation,
    canonical_features,
)
from .geometry import GeometryResult, verify_geometry
from .matching import MatchRecord, MatchSet, match_descriptors
from .scoring import raw_score, score_components
from .types import CanonicalLocalFeature, LocalFeatureRepresentation

__all__ = [
    "PROTOCOL_NAME",
    "CanonicalLocalFeature",
    "DetectorOnlyAdapter",
    "DetectorOnlyProtocolConfig",
    "GeometryResult",
    "LocalFeatureRepresentation",
    "MatchRecord",
    "MatchSet",
    "adapt_detector_result",
    "build_representation",
    "canonical_features",
    "match_descriptors",
    "raw_score",
    "score_components",
    "verify_geometry",
]
