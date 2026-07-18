"""PPI-aware orientation support for HarrisZ+ v4."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .orientation import (
    OrientationResult,
    _validate_config_policy,
    build_orientation_histogram,
    parabolic_peak_angle,
    signed_gradient_maps,
    smooth_circular_histogram,
)
from .ppi_aware_v4 import PpiAwareRuntimeConfig


def orientation_from_gradients_v4(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    *,
    x: float,
    y: float,
    scale_index: int,
    config: PpiAwareRuntimeConfig,
) -> OrientationResult:
    """Assign the unchanged v3 angle using an explicitly scaled support radius."""

    weighting_sigma = config.orientation_weighting_sigma(scale_index)
    radius = config.orientation_radius_pixels(scale_index)
    histogram, sample_count = build_orientation_histogram(
        gradient_x,
        gradient_y,
        x=float(x),
        y=float(y),
        weighting_sigma=weighting_sigma,
        radius=radius,
        bins=int(config.orientation_bins),
    )
    smoothed = smooth_circular_histogram(
        histogram,
        passes=int(config.orientation_histogram_smoothing_passes),
    )
    angle, peak_bin, peak_offset, peak_value = parabolic_peak_angle(smoothed)
    return OrientationResult(
        angle_degrees=angle,
        histogram=smoothed,
        peak_bin=peak_bin,
        peak_offset=peak_offset,
        peak_value=peak_value,
        sample_count=sample_count,
        radius_pixels=radius,
        weighting_sigma=weighting_sigma,
    )


def assign_orientations_v4_with_diagnostics(
    image: np.ndarray,
    keypoints: Iterable[object],
    config: PpiAwareRuntimeConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign one deterministic signed-gradient orientation per keypoint."""

    records = tuple(keypoints)
    _validate_config_policy(config)
    gradient_kernel = tuple(config.orientation_gradient_kernel)
    border_mode = str(config.orientation_border_mode)
    gradient_x, gradient_y = signed_gradient_maps(
        image,
        gradient_kernel=gradient_kernel,
        border_mode=border_mode,
    )
    results = [
        orientation_from_gradients_v4(
            gradient_x,
            gradient_y,
            x=float(getattr(keypoint, "x")),
            y=float(getattr(keypoint, "y")),
            scale_index=int(getattr(keypoint, "scale_index")),
            config=config,
        )
        for keypoint in records
    ]
    angles = np.asarray(
        [result.angle_degrees for result in results],
        dtype=np.float32,
    )
    radius_table = {
        str(index): {
            "radius_pixels": config.orientation_radius_pixels(index),
            "radius_mm": (
                config.orientation_radius_pixels(index)
                * 25.4
                / config.manifest_ppi
            ),
            "histogram_window_size": (
                2 * config.orientation_radius_pixels(index) + 1
            ),
            "weighting_sigma_pixels": config.orientation_weighting_sigma(index),
        }
        for index in config.scale_indices
    }
    return angles, {
        "orientation_method": "sift_style_dominant_signed_gradient",
        "orientation_count": int(angles.size),
        "orientation_bins": int(config.orientation_bins),
        "orientation_gradient_kernel": list(gradient_kernel),
        "orientation_border_mode": border_mode,
        "orientation_gaussian_sigma_factor": float(
            config.orientation_gaussian_sigma_factor
        ),
        "orientation_radius_factor_of_weighting_sigma": float(
            config.orientation_radius_factor
        ),
        "orientation_radius_policy": (
            "reference_rounded_radius_times_manifest_ppi_over_1000"
        ),
        "orientation_radius_by_scale": radius_table,
        "orientation_histogram_interpolation": "linear_circular",
        "orientation_histogram_smoothing": (
            "repeated_three_bin_circular_mean"
        ),
        "orientation_histogram_smoothing_passes": int(
            config.orientation_histogram_smoothing_passes
        ),
        "orientation_peak_interpolation": (
            "three_sample_parabola_clipped_to_half_bin"
        ),
        "orientation_tie_policy": "lowest_interpolated_angle",
        "orientation_180_degree_policy": "full_signed_gradient_no_random_flip",
        "orientation_scale_source": (
            "ppi_scaled_output_integration_sigma_with_explicit_scaled_radius"
        ),
        "empty_histogram_angle_degrees": 0.0,
        "orientation_sample_count_min": (
            min(result.sample_count for result in results) if results else 0
        ),
        "orientation_sample_count_max": (
            max(result.sample_count for result in results) if results else 0
        ),
    }


def assign_orientations_v4(
    image: np.ndarray,
    keypoints: Iterable[object],
    config: PpiAwareRuntimeConfig,
) -> np.ndarray:
    angles, _ = assign_orientations_v4_with_diagnostics(
        image, keypoints, config
    )
    return angles


__all__ = [
    "assign_orientations_v4",
    "assign_orientations_v4_with_diagnostics",
    "orientation_from_gradients_v4",
]
