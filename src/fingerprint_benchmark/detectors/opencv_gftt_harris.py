"""Classical OpenCV GFTT-Harris detector and public detector-only adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter_ns
from typing import Any, Mapping

import cv2
import numpy as np

from fingerprint_benchmark.local_features.detector_only import (
    DetectorOnlyAdapter,
    DetectorOnlyProtocolConfig,
)

from .types import DetectedPoint, DetectorResult


DETECTOR_NAME = "opencv_gftt_harris"
DETECTOR_VERSION = "opencv-gftt-harris-v1"
METHOD_NAME = "opencv_gftt_harris_rootsift_geometric"
METHOD_VERSION = "opencv-gftt-harris-rootsift-geometric-v1"


@dataclass(frozen=True, slots=True)
class OpenCVHarrisConfig:
    """Explicit parameters passed unchanged to OpenCV corner selection."""

    max_corners: int = 3000
    quality_level: float = 0.01
    min_distance: float = 5.0
    block_size: int = 3
    gradient_size: int = 3
    harris_k: float = 0.04

    def __post_init__(self) -> None:
        if self.max_corners <= 0:
            raise ValueError("max_corners must be positive.")
        if not 0.0 < self.quality_level <= 1.0:
            raise ValueError("quality_level must be in (0, 1].")
        if self.min_distance < 0.0:
            raise ValueError("min_distance must be non-negative.")
        if self.block_size <= 0 or self.block_size % 2 == 0:
            raise ValueError("block_size must be a positive odd integer.")
        if self.gradient_size <= 0 or self.gradient_size % 2 == 0:
            raise ValueError("gradient_size must be a positive odd integer.")
        if self.gradient_size > self.block_size:
            raise ValueError("gradient_size must not exceed block_size.")
        if not np.isfinite(self.harris_k) or self.harris_k <= 0.0:
            raise ValueError("harris_k must be finite and positive.")

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


class OpenCVGFTTHarrisDetector:
    """Thin wrapper that leaves the complete corner-selection policy to OpenCV."""

    detector_name = DETECTOR_NAME
    detector_version = DETECTOR_VERSION

    def __init__(self, config: OpenCVHarrisConfig | None = None) -> None:
        self.config = config or OpenCVHarrisConfig()

    def detect(
        self,
        image: np.ndarray,
        image_metadata: Mapping[str, object],
        mask: np.ndarray | None = None,
    ) -> DetectorResult:
        source = np.asarray(image)
        if source.ndim != 2 or source.size == 0:
            raise ValueError("OpenCV GFTT-Harris requires a non-empty grayscale image.")
        if source.dtype not in (np.uint8, np.float32):
            raise ValueError("OpenCV GFTT-Harris requires a uint8 or float32 image.")
        if mask is not None:
            active_mask = np.asarray(mask)
            if active_mask.shape != source.shape or active_mask.dtype != np.uint8:
                raise ValueError("mask must be uint8 and have the same shape as the image.")

        parameters = self.config.as_dict()
        started = perf_counter_ns()
        corners = cv2.goodFeaturesToTrack(
            image=source,
            maxCorners=int(self.config.max_corners),
            qualityLevel=float(self.config.quality_level),
            minDistance=float(self.config.min_distance),
            mask=mask,
            blockSize=int(self.config.block_size),
            gradientSize=int(self.config.gradient_size),
            useHarrisDetector=True,
            k=float(self.config.harris_k),
        )
        detector_time_ms = (perf_counter_ns() - started) / 1_000_000.0

        # goodFeaturesToTrack exposes its ranked coordinates but not response
        # magnitudes.  Zero is an honest sentinel; tuple order is OpenCV's rank.
        points: tuple[DetectedPoint, ...]
        if corners is None:
            points = ()
        else:
            coordinates = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
            points = tuple(
                DetectedPoint(
                    x=float(x),
                    y=float(y),
                    response=0.0,
                    detector_metadata={"opencv_rank": int(rank)},
                )
                for rank, (x, y) in enumerate(coordinates)
            )
        metadata: dict[str, object] = {
            "opencv_version": str(cv2.__version__),
            "opencv_parameters": dict(parameters),
            "response_semantics": "unavailable_from_goodFeaturesToTrack_zero_sentinel",
            "ranking_semantics": "opencv_return_order",
            "image_metadata": dict(image_metadata),
        }
        return DetectorResult(
            points=points,
            detector_name=self.detector_name,
            detector_version=self.detector_version,
            detector_config=parameters,
            diagnostics={
                "corner_count": len(points),
                "corners_is_none": corners is None,
                "mask_supplied": mask is not None,
                "image_shape": [int(source.shape[0]), int(source.shape[1])],
                "image_dtype": str(source.dtype),
            },
            detector_time_ms=detector_time_ms,
            metadata=metadata,
        )


class OpenCVGFTTHarrisRootSIFTGeometricAdapter(DetectorOnlyAdapter):
    """Public Harris -> detector_only_v1 -> RootSIFT -> geometry method."""

    def __init__(
        self,
        detector_config: OpenCVHarrisConfig | None = None,
        protocol_config: DetectorOnlyProtocolConfig | None = None,
    ) -> None:
        super().__init__(
            detector=OpenCVGFTTHarrisDetector(detector_config),
            config=protocol_config or DetectorOnlyProtocolConfig(),
            method_name=METHOD_NAME,
            method_version=METHOD_VERSION,
        )


# Concise public spellings.
OpenCVHarrisDetector = OpenCVGFTTHarrisDetector
OpenCVHarrisRootSIFTGeometricAdapter = OpenCVGFTTHarrisRootSIFTGeometricAdapter


__all__ = [
    "DETECTOR_NAME",
    "DETECTOR_VERSION",
    "METHOD_NAME",
    "METHOD_VERSION",
    "OpenCVHarrisConfig",
    "OpenCVGFTTHarrisDetector",
    "OpenCVHarrisDetector",
    "OpenCVGFTTHarrisRootSIFTGeometricAdapter",
    "OpenCVHarrisRootSIFTGeometricAdapter",
]
