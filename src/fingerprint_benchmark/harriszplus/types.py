"""Small immutable records shared by HarrisZ+ detector backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class ScaleSpec:
    scale_index: int
    nominal_sigma: float
    working_image_scale: float
    working_sigma: float
    output_sigma: float
    working_integration_sigma: float
    output_integration_sigma: float
    effective_support_diameter: float
    keypoint_size: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SelectionCandidate:
    """Backend-neutral candidate in original-image coordinates."""

    x: float
    y: float
    response: float
    scale_index: int
    source_index: int


@dataclass(frozen=True, slots=True)
class DetectedKeypoint:
    """A HarrisZ+ keypoint after all detector-side selection stages.

    ``sigma`` is the final differentiation sigma.  ``integration_sigma`` is
    explicit because OpenCV ``KeyPoint.size`` is twice that output scale.
    """

    x: float
    y: float
    response: float
    scale_index: int
    sigma: float
    integration_sigma: float
    effective_support_diameter: float
    size: float
    source_index: int

    @property
    def pt(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def differentiation_sigma(self) -> float:
        return self.sigma

    @property
    def output_integration_sigma(self) -> float:
        return self.integration_sigma

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DetectorResult:
    backend: str
    keypoints: tuple[DetectedKeypoint, ...]
    diagnostics: Mapping[str, Any]
    timings: Mapping[str, float]
    response_maps: Mapping[int, np.ndarray] | None = None

    @property
    def count(self) -> int:
        return len(self.keypoints)

    def as_dict(self, *, include_response_maps: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "backend": self.backend,
            "keypoints": [keypoint.as_dict() for keypoint in self.keypoints],
            "diagnostics": dict(self.diagnostics),
            "timings": dict(self.timings),
        }
        if include_response_maps:
            payload["response_maps"] = self.response_maps
        return payload


def make_scale_spec(config: Any, scale_index: int) -> ScaleSpec:
    """Build a serializable scale record without coupling types to config."""

    return ScaleSpec(
        scale_index=scale_index,
        nominal_sigma=config.nominal_sigma(scale_index),
        working_image_scale=config.working_image_scale(scale_index),
        working_sigma=config.working_sigma(scale_index),
        output_sigma=config.output_sigma(scale_index),
        working_integration_sigma=config.working_integration_sigma(scale_index),
        output_integration_sigma=config.output_integration_sigma(scale_index),
        effective_support_diameter=config.effective_gaussian_support_diameter(scale_index),
        keypoint_size=config.keypoint_size(scale_index),
    )
