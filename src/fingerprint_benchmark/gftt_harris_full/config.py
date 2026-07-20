"""Immutable, explicit configuration for the full Harris matcher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fingerprint_benchmark.detectors.opencv_gftt_harris import OpenCVHarrisConfig
from fingerprint_benchmark.hashing import stable_config_hash
from fingerprint_benchmark.local_features.detector_only import DetectorOnlyProtocolConfig
from fingerprint_benchmark.local_features.orientation import ORIENTATION_POLICY


CONFIG_SCHEMA_VERSION = "gftt-harris-rootsift-geometric-config-v1"


@dataclass(frozen=True, slots=True)
class GFTTHarrisRootSIFTGeometricConfig:
    """Every algorithmic value frozen by the Joint-500 Harris run.

    Conversion methods state every downstream value explicitly.  In
    particular, the v1 method never inherits algorithm choices from mutable
    defaults in the detector-only study.
    """

    max_corners: int = 3000
    quality_level: float = 0.01
    min_distance: float = 5.0
    block_size: int = 3
    gradient_size: int = 3
    harris_k: float = 0.04
    reference_ppi: float = 1000.0
    support_size_reference_px: float = 16.0
    maximum_keypoints: int = 3000
    descriptor: str = "rootsift"
    orientation_policy: str = ORIENTATION_POLICY
    matching_mode: str = "mutual"
    lowe_ratio: float = 0.75
    geometry_model: str = "affine_partial_2d"
    ransac_threshold_reference_px: float = 3.0
    minimum_descriptors: int = 2
    minimum_geometry_matches: int = 3
    ransac_confidence: float = 0.99
    ransac_max_iterations: int = 2000
    ransac_refine_iterations: int = 10
    rng_seed: int = 0
    opencv_threads: int = 16
    opencv_optimized: bool = True

    def __post_init__(self) -> None:
        # The protected constructors are the validation authority.  Values are
        # passed explicitly so validation itself cannot import a default.
        self.to_detector_config()
        self.to_pipeline_config()

    def to_detector_config(self) -> OpenCVHarrisConfig:
        return OpenCVHarrisConfig(
            max_corners=self.max_corners,
            quality_level=self.quality_level,
            min_distance=self.min_distance,
            block_size=self.block_size,
            gradient_size=self.gradient_size,
            harris_k=self.harris_k,
        )

    def to_pipeline_config(self) -> DetectorOnlyProtocolConfig:
        return DetectorOnlyProtocolConfig(
            reference_ppi=self.reference_ppi,
            support_size_reference_px=self.support_size_reference_px,
            maximum_keypoints=self.maximum_keypoints,
            descriptor=self.descriptor,
            orientation_policy=self.orientation_policy,
            matching_mode=self.matching_mode,
            lowe_ratio=self.lowe_ratio,
            geometry_model=self.geometry_model,
            ransac_threshold_reference_px=self.ransac_threshold_reference_px,
            minimum_descriptors=self.minimum_descriptors,
            minimum_geometry_matches=self.minimum_geometry_matches,
            ransac_confidence=self.ransac_confidence,
            ransac_max_iterations=self.ransac_max_iterations,
            ransac_refine_iterations=self.ransac_refine_iterations,
            rng_seed=self.rng_seed,
            opencv_threads=self.opencv_threads,
            opencv_optimized=self.opencv_optimized,
        )

    def algorithm_config(self) -> dict[str, Any]:
        """Canonical score-affecting values; descriptive metadata is excluded."""

        detector = self.to_detector_config().as_dict()
        pipeline = self.to_pipeline_config().as_dict()
        return {
            "detector": {
                **detector,
                "mask": None,
                "use_harris_detector": True,
            },
            "pipeline": {
                **pipeline,
                "normalize_coordinates_by_ppi": True,
                "score_mode": "geometric_inlier_count",
            },
        }

    @property
    def config_hash(self) -> str:
        return stable_config_hash(self.algorithm_config())


def frozen_config() -> GFTTHarrisRootSIFTGeometricConfig:
    """Return v1 without relying on any dataclass or downstream defaults."""

    return GFTTHarrisRootSIFTGeometricConfig(
        max_corners=3000,
        quality_level=0.01,
        min_distance=5.0,
        block_size=3,
        gradient_size=3,
        harris_k=0.04,
        reference_ppi=1000.0,
        support_size_reference_px=16.0,
        maximum_keypoints=3000,
        descriptor="rootsift",
        orientation_policy="common_dominant_gradient_v1",
        matching_mode="mutual",
        lowe_ratio=0.75,
        geometry_model="affine_partial_2d",
        ransac_threshold_reference_px=3.0,
        minimum_descriptors=2,
        minimum_geometry_matches=3,
        ransac_confidence=0.99,
        ransac_max_iterations=2000,
        ransac_refine_iterations=10,
        rng_seed=0,
        opencv_threads=16,
        opencv_optimized=True,
    )


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "GFTTHarrisRootSIFTGeometricConfig",
    "frozen_config",
]
