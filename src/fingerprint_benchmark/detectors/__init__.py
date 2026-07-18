"""Detector contracts and repository-native detector implementations."""

from typing import Any

from .types import DetectedPoint, Detector, DetectorResult

__all__ = [
    "DetectedPoint",
    "Detector",
    "DetectorResult",
    "OpenCVHarrisConfig",
    "OpenCVGFTTHarrisDetector",
    "OpenCVHarrisDetector",
    "OpenCVGFTTHarrisRootSIFTGeometricAdapter",
    "OpenCVHarrisRootSIFTGeometricAdapter",
]


def __getattr__(name: str) -> Any:
    """Load concrete detectors lazily so the generic protocol stays acyclic."""

    if name in {
        "OpenCVHarrisConfig",
        "OpenCVGFTTHarrisDetector",
        "OpenCVHarrisDetector",
        "OpenCVGFTTHarrisRootSIFTGeometricAdapter",
        "OpenCVHarrisRootSIFTGeometricAdapter",
    }:
        from . import opencv_gftt_harris

        return getattr(opencv_gftt_harris, name)
    raise AttributeError(name)
