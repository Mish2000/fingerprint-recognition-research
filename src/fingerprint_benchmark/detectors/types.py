"""Method-neutral detector contracts for detector-only research."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class DetectedPoint:
    """One ranked detector output in source-image coordinates.

    Native scale and angle are retained only as provenance.  The
    ``detector_only_v1`` protocol deliberately ignores both fields.
    """

    x: float
    y: float
    response: float
    detector_scale: float | None = None
    detector_angle: float | None = None
    detector_metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not all(math.isfinite(float(value)) for value in (self.x, self.y, self.response)):
            raise ValueError("Detected point coordinates and response must be finite.")
        for name, value in (
            ("detector_scale", self.detector_scale),
            ("detector_angle", self.detector_angle),
        ):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when present.")


@dataclass(frozen=True, slots=True)
class DetectorResult:
    """Detector-only output with no descriptors, matching, or decisions."""

    points: tuple[DetectedPoint, ...]
    detector_name: str
    detector_version: str
    detector_config: Mapping[str, object] = field(default_factory=dict)
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    detector_time_ms: float = 0.0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.detector_name.strip() or not self.detector_version.strip():
            raise ValueError("Detector name and version must be non-empty.")
        if not math.isfinite(float(self.detector_time_ms)) or self.detector_time_ms < 0.0:
            raise ValueError("Detector time must be finite and non-negative.")
        if not isinstance(self.points, tuple):
            object.__setattr__(self, "points", tuple(self.points))
        if not all(isinstance(point, DetectedPoint) for point in self.points):
            raise TypeError("DetectorResult.points must contain DetectedPoint records.")

    @property
    def count(self) -> int:
        return len(self.points)

    @property
    def keypoints(self) -> tuple[DetectedPoint, ...]:
        """Compatibility spelling for detector implementations using keypoints."""

        return self.points

    @property
    def name(self) -> str:
        return self.detector_name

    @property
    def version(self) -> str:
        return self.detector_version

    @property
    def config(self) -> Mapping[str, object]:
        return self.detector_config

    @property
    def timings(self) -> Mapping[str, float]:
        return {"detector_time_ms": float(self.detector_time_ms)}


@runtime_checkable
class Detector(Protocol):
    """Minimal interface implemented by every detector under comparison."""

    detector_name: str
    detector_version: str

    def detect(
        self,
        image: np.ndarray,
        image_metadata: Mapping[str, object],
        mask: np.ndarray | None = None,
    ) -> DetectorResult:
        ...
