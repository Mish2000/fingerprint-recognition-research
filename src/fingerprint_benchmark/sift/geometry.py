"""PPI-normalized affine verification and residual diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .config import SiftGeometricConfig
from .extractor import SiftRepresentation
from .matching import MatchRecord


@dataclass(frozen=True)
class GeometryResult:
    success: bool
    failure_reason: str | None
    transform: np.ndarray | None
    inlier_mask: np.ndarray
    inlier_count: int
    outlier_count: int
    inlier_ratio: float
    diagnostics: dict[str, object]


def verify_geometry(
    representation_a: SiftRepresentation,
    representation_b: SiftRepresentation,
    matches: tuple[MatchRecord, ...],
    config: SiftGeometricConfig,
) -> GeometryResult:
    submitted = len(matches)
    if submitted < int(config.minimum_geometry_matches):
        return _failure("insufficient_geometry_matches", submitted, config)
    points_a = np.asarray([representation_a.points[m.index_a] for m in matches], dtype=np.float32)
    points_b = np.asarray([representation_b.points[m.index_b] for m in matches], dtype=np.float32)
    if config.normalize_coordinates_by_ppi:
        points_a = points_a * (float(config.reference_ppi) / representation_a.ppi)
        points_b = points_b * (float(config.reference_ppi) / representation_b.ppi)
    cv2.setRNGSeed(int(config.rng_seed))
    kwargs = {
        "method": cv2.RANSAC,
        "ransacReprojThreshold": float(config.ransac_threshold_at_reference_ppi),
        "maxIters": int(config.ransac_max_iterations),
        "confidence": float(config.ransac_confidence),
        "refineIters": int(config.ransac_refine_iterations),
    }
    try:
        if config.geometry_model == "affine_full_2d":
            transform, raw_mask = cv2.estimateAffine2D(points_a, points_b, **kwargs)
        elif config.geometry_model == "affine_partial_2d":
            transform, raw_mask = cv2.estimateAffinePartial2D(points_a, points_b, **kwargs)
        else:
            raise ValueError(f"Unsupported geometry model: {config.geometry_model!r}.")
    except cv2.error:
        return _failure("opencv_geometry_error", submitted, config)
    if transform is None or raw_mask is None or not np.isfinite(transform).all():
        return _failure("model_estimation_failure", submitted, config)
    inlier_mask = raw_mask.reshape(-1).astype(bool)
    if inlier_mask.size != submitted:
        return _failure("invalid_inlier_mask", submitted, config)
    inlier_count = int(inlier_mask.sum())
    outlier_count = submitted - inlier_count
    inlier_ratio = float(inlier_count / submitted) if submitted else 0.0
    homogeneous = np.column_stack([points_a, np.ones(submitted, dtype=np.float32)])
    predicted = homogeneous @ np.asarray(transform, dtype=np.float64).T
    residuals = np.linalg.norm(predicted - points_b, axis=1)
    inlier_residuals = residuals[inlier_mask]
    residual_summary = _summary(inlier_residuals)
    linear = np.asarray(transform, dtype=np.float64)[:, :2]
    determinant = float(np.linalg.det(linear))
    scale = float(math.sqrt(abs(determinant)))
    rotation = float(math.degrees(math.atan2(linear[1, 0], linear[0, 0])))
    destination_pixel_scale = representation_b.ppi / float(config.reference_ppi)
    return GeometryResult(
        success=True,
        failure_reason=None,
        transform=np.asarray(transform, dtype=np.float64),
        inlier_mask=inlier_mask,
        inlier_count=inlier_count,
        outlier_count=outlier_count,
        inlier_ratio=inlier_ratio,
        diagnostics={
            "geometry_model": config.geometry_model,
            "geometry_success": True,
            "geometry_failure_reason": None,
            "geometric_inlier_count": inlier_count,
            "geometric_outlier_count": outlier_count,
            "inlier_ratio": inlier_ratio,
            "estimated_transform": np.asarray(transform, dtype=float).tolist(),
            "transform_determinant": determinant,
            "scale_estimate": scale,
            "rotation_degrees": rotation,
            "translation_x_reference_pixels": float(transform[0, 2]),
            "translation_y_reference_pixels": float(transform[1, 2]),
            "residual_reference_pixels": residual_summary,
            "residual_destination_pixels": {
                key: (None if value is None else float(value) * destination_pixel_scale)
                for key, value in residual_summary.items()
            },
            "coordinate_normalization": "ppi_to_reference" if config.normalize_coordinates_by_ppi else "none",
            "reference_ppi": float(config.reference_ppi),
            "ransac_threshold_reference_pixels": float(config.ransac_threshold_at_reference_ppi),
            "ransac_confidence": float(config.ransac_confidence),
            "ransac_max_iterations": int(config.ransac_max_iterations),
            "ransac_refine_iterations": int(config.ransac_refine_iterations),
            "ransac_iterations_observed": None,
        },
    )


def _failure(reason: str, submitted: int, config: SiftGeometricConfig) -> GeometryResult:
    return GeometryResult(
        success=False,
        failure_reason=reason,
        transform=None,
        inlier_mask=np.zeros(submitted, dtype=bool),
        inlier_count=0,
        outlier_count=submitted,
        inlier_ratio=0.0,
        diagnostics={
            "geometry_model": config.geometry_model,
            "geometry_success": False,
            "geometry_failure_reason": reason,
            "geometric_inlier_count": 0,
            "geometric_outlier_count": submitted,
            "inlier_ratio": 0.0,
            "estimated_transform": None,
            "transform_determinant": None,
            "scale_estimate": None,
            "rotation_degrees": None,
            "translation_x_reference_pixels": None,
            "translation_y_reference_pixels": None,
            "residual_reference_pixels": _summary(np.asarray([], dtype=float)),
            "residual_destination_pixels": _summary(np.asarray([], dtype=float)),
            "coordinate_normalization": "ppi_to_reference" if config.normalize_coordinates_by_ppi else "none",
            "reference_ppi": float(config.reference_ppi),
            "ransac_threshold_reference_pixels": float(config.ransac_threshold_at_reference_ppi),
            "ransac_confidence": float(config.ransac_confidence),
            "ransac_max_iterations": int(config.ransac_max_iterations),
            "ransac_refine_iterations": int(config.ransac_refine_iterations),
            "ransac_iterations_observed": None,
        },
    )


def _summary(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"mean": None, "median": None, "p95": None, "maximum": None}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }
