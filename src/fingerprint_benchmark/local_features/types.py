"""Shared local-feature representation types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class CanonicalLocalFeature:
    """A detector location after common support and orientation assignment."""

    x: float
    y: float
    support_size: float
    angle: float


@dataclass(frozen=True)
class LocalFeatureRepresentation:
    """Detector-neutral points and descriptors consumed by matching and geometry."""

    points: np.ndarray
    sizes: np.ndarray
    angles: np.ndarray
    responses: np.ndarray
    octaves: np.ndarray
    class_ids: np.ndarray
    descriptors: np.ndarray
    width: int
    height: int
    ppi: float
    metadata: dict[str, Any]

    @property
    def keypoint_count(self) -> int:
        return int(self.points.shape[0])


# Generic spelling for consumers migrating from the historical SIFT type name.
FeatureRepresentation = LocalFeatureRepresentation


__all__ = [
    "CanonicalLocalFeature",
    "FeatureRepresentation",
    "LocalFeatureRepresentation",
]
