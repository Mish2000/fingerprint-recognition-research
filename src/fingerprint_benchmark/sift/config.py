"""Frozen, validated configuration for the single public SIFT method."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any


METHOD_NAME = "sift_geometric"
METHOD_VERSION = "sift-geometric-v1"
REPRESENTATION_VERSION = "sift-geometric-representation-v1"
CONFIG_SCHEMA_VERSION = "sift-geometric-config-schema-v1"

DESCRIPTOR_MODES = ("standard", "rootsift")
MATCHING_MODES = ("one_way", "bidirectional_union", "mutual")
GEOMETRY_MODELS = ("affine_full_2d", "affine_partial_2d")
SCORE_MODES = (
    "geometric_inlier_count",
    "geometric_inlier_ratio",
    "inliers_over_min_keypoints",
    "inliers_times_inlier_ratio_times_log1p_matches",
)
IMAGE_POLICIES = ("native", "reference_reproduction")
MASK_MODES = ("none", "valid_region")


@dataclass(frozen=True)
class SiftGeometricConfig:
    schema_version: str = CONFIG_SCHEMA_VERSION
    image_policy: str = "native"
    mask_mode: str = "none"
    descriptor_mode: str = "standard"
    matching_mode: str = "mutual"
    geometry_model: str = "affine_partial_2d"
    score_mode: str = "inliers_times_inlier_ratio_times_log1p_matches"
    nfeatures: int = 3000
    n_octave_layers: int = 3
    contrast_threshold: float = 0.04
    edge_threshold: float = 10.0
    sigma: float = 1.6
    lowe_ratio: float = 0.75
    minimum_descriptors: int = 2
    minimum_geometry_matches: int = 3
    ransac_threshold_at_reference_ppi: float = 3.0
    ransac_confidence: float = 0.99
    ransac_max_iterations: int = 2000
    ransac_refine_iterations: int = 10
    normalize_coordinates_by_ppi: bool = True
    reference_ppi: float = 1000.0
    rng_seed: int = 0
    opencv_threads: int = 16
    opencv_optimized: bool = True
    valid_region_black_threshold: int = 10
    valid_region_close_kernel_at_reference_ppi: int = 31
    valid_region_erode_at_reference_ppi: int = 5
    valid_region_min_coverage: float = 0.01
    reference_target_size: int = 768
    reference_clahe_clip: float = 2.0
    reference_clahe_grid_x: int = 8
    reference_clahe_grid_y: int = 8

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"Unsupported config schema: {self.schema_version!r}.")
        _choice("image_policy", self.image_policy, IMAGE_POLICIES)
        _choice("mask_mode", self.mask_mode, MASK_MODES)
        _choice("descriptor_mode", self.descriptor_mode, DESCRIPTOR_MODES)
        _choice("matching_mode", self.matching_mode, MATCHING_MODES)
        _choice("geometry_model", self.geometry_model, GEOMETRY_MODELS)
        _choice("score_mode", self.score_mode, SCORE_MODES)
        if self.nfeatures <= 0 or self.minimum_descriptors < 2:
            raise ValueError("nfeatures must be positive and minimum_descriptors must be at least two.")
        if self.minimum_geometry_matches < 3:
            raise ValueError("Affine verification requires at least three matches.")
        if not 0.0 < self.lowe_ratio < 1.0:
            raise ValueError("lowe_ratio must be between zero and one.")
        if not 0.0 < self.ransac_confidence < 1.0:
            raise ValueError("ransac_confidence must be between zero and one.")
        if self.ransac_threshold_at_reference_ppi <= 0 or self.reference_ppi <= 0:
            raise ValueError("RANSAC threshold and reference PPI must be positive.")
        if self.image_policy == "native" and not self.normalize_coordinates_by_ppi:
            raise ValueError("Native-resolution geometry must normalize coordinates by PPI.")
        if not 0.0 <= self.valid_region_min_coverage <= 1.0:
            raise ValueError("valid_region_min_coverage must be in [0, 1].")
        if self.opencv_threads <= 0:
            raise ValueError("opencv_threads must be positive and explicit.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def changed(self, **changes: Any) -> "SiftGeometricConfig":
        return replace(self, **changes)

    @classmethod
    def from_json(cls, path: Path) -> "SiftGeometricConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("SIFT config JSON must contain an object.")
        return cls(**payload)


def _choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}; got {value!r}.")
