"""Frozen correspondence with the authoritative Joint-500 Harris method."""

from __future__ import annotations

from typing import Any

from .config import GFTTHarrisRootSIFTGeometricConfig, frozen_config


PARENT_METHOD_NAME = "opencv_gftt_harris_rootsift_geometric"
PARENT_METHOD_VERSION = "opencv-gftt-harris-rootsift-geometric-v1"
PARENT_RUN_CONFIG_HASH = "cf362a9bcf88c1308481a4e88d545842db1213f4aa99e7148f77293a1ede74f1"
PARENT_PROTOCOL_SHA256 = "4d53ba3466524f6a0399e57f62edc1bac58fb2bb425e18bdbd4ef95373e7ec23"

# Literal values from the authoritative Joint-500 run_metadata.json.  They are
# intentionally independent of detector-only constructors.
HISTORICAL_DETECTOR_CONFIG: dict[str, int | float] = {
    "max_corners": 3000,
    "quality_level": 0.01,
    "min_distance": 5.0,
    "block_size": 3,
    "gradient_size": 3,
    "harris_k": 0.04,
}
HISTORICAL_PIPELINE_CONFIG: dict[str, Any] = {
    "reference_ppi": 1000.0,
    "support_size_reference_px": 16.0,
    "maximum_keypoints": 3000,
    "descriptor": "rootsift",
    "orientation_policy": "common_dominant_gradient_v1",
    "matching_mode": "mutual",
    "lowe_ratio": 0.75,
    "geometry_model": "affine_partial_2d",
    "ransac_threshold_reference_px": 3.0,
    "minimum_descriptors": 2,
    "minimum_geometry_matches": 3,
    "ransac_confidence": 0.99,
    "ransac_max_iterations": 2000,
    "ransac_refine_iterations": 10,
    "rng_seed": 0,
    "opencv_threads": 16,
    "opencv_optimized": True,
}


def equivalence_mismatches(
    config: GFTTHarrisRootSIFTGeometricConfig,
) -> dict[str, dict[str, Any]]:
    actual = {
        "detector": config.to_detector_config().as_dict(),
        "pipeline": config.to_pipeline_config().as_dict(),
    }
    expected = {
        "detector": HISTORICAL_DETECTOR_CONFIG,
        "pipeline": HISTORICAL_PIPELINE_CONFIG,
    }
    return {
        section: {"expected": expected[section], "actual": actual[section]}
        for section in expected
        if actual[section] != expected[section]
    }


def assert_v1_equivalence(
    config: GFTTHarrisRootSIFTGeometricConfig | None = None,
) -> None:
    active = config if config is not None else frozen_config()
    mismatches = equivalence_mismatches(active)
    if mismatches:
        raise ValueError(
            "gftt-harris-rootsift-geometric-v1 only accepts the frozen "
            f"Joint-500 algorithm values; mismatches={mismatches!r}"
        )


__all__ = [
    "HISTORICAL_DETECTOR_CONFIG",
    "HISTORICAL_PIPELINE_CONFIG",
    "PARENT_METHOD_NAME",
    "PARENT_METHOD_VERSION",
    "PARENT_PROTOCOL_SHA256",
    "PARENT_RUN_CONFIG_HASH",
    "assert_v1_equivalence",
    "equivalence_mismatches",
]
